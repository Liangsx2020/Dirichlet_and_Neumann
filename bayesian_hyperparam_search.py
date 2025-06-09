import os
import time
import torch
import warnings
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import json

from GNP.problems import *
from GNP.solver import GMRES
from GNP.preconditioners import *
from GNP.nn import ResGCN

# Bayesian optimization imports
try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args
    from skopt.plots import plot_convergence, plot_objective
    print("Using scikit-optimize for Bayesian optimization")
except ImportError:
    print("Installing scikit-optimize...")
    os.system("pip install scikit-optimize")
    from skopt import gp_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args
    from skopt.plots import plot_convergence, plot_objective


class GNPObjective:
    """Objective function for Bayesian optimization"""
    
    def __init__(self, n=16, max_train_time=200):
        self.n = n
        self.max_train_time = max_train_time
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Pre-compute baseline once
        print(f"Computing baseline for n={n}...")
        A = gen_2d_poisson_matrix(n)
        b = build_rhs_2d(n)
        solver = GMRES()
        
        try:
            start_time = time.time()
            x_baseline, self.baseline_iters, _, _, _ = solver.solve(
                A, b, M=None, restart=15, max_iters=2000, 
                timeout=60, rtol=1e-8, progress_bar=False)
            self.baseline_time = time.time() - start_time
            self.baseline_success = True
            print(f"Baseline: {self.baseline_iters} iters, {self.baseline_time:.2f}s")
        except:
            self.baseline_time = np.inf
            self.baseline_iters = 2000
            self.baseline_success = False
            print("Baseline failed!")
        
        # Store problem for reuse
        self.A = A
        self.b = b
        self.solver = solver
        
        # Track best results
        self.best_score = -np.inf
        self.best_config = None
        self.evaluation_history = []
    
    def __call__(self, params):
        """Objective function to minimize (negative performance score)"""
        
        # Unpack parameters
        training_data, m, embed, num_layers, lr_log, drop_rate = params
        lr = 10 ** lr_log  # Convert from log scale
        
        config = {
            'training_data': training_data,
            'm': int(m),
            'embed': int(embed),
            'num_layers': int(num_layers),
            'lr': lr,
            'drop_rate': drop_rate,
            'hidden': int(embed * 2),  # Adaptive hidden size
            'epochs': min(400, int(200 + embed * 2)),  # Adaptive epochs
            'batch_size': 8,
            'weight_decay': 1e-5,
            'scale_input': True
        }
        
        print(f"\nEvaluating: {config}")
        
        try:
            # Train GNP
            net = ResGCN(self.A, config['num_layers'], config['embed'], 
                        config['hidden'], config['drop_rate'], self.device, 
                        scale_input=config['scale_input'], dtype=torch.float32).to(self.device)
            
            optimizer = torch.optim.AdamW(net.parameters(), lr=config['lr'], 
                                         weight_decay=config['weight_decay'])
            
            M = GNP(self.A, config['training_data'], config['m'], net, self.device)
            
            train_start = time.time()
            hist_loss, best_loss, best_epoch, _ = M.train(
                config['batch_size'], 1, config['epochs'], optimizer, None,
                num_workers=0, progress_bar=False)
            train_time = time.time() - train_start
            
            if train_time > self.max_train_time:
                print(f"Training timeout: {train_time:.1f}s")
                return 10.0  # High penalty for timeout
                
            # Test GNP solve
            warnings.filterwarnings('error')
            solve_start = time.time()
            x_gnp, iters_gnp, _, _, _ = self.solver.solve(
                self.A, self.b, M=M, restart=15, max_iters=2000,
                timeout=60, rtol=1e-8, progress_bar=False)
            solve_time = time.time() - solve_start
            warnings.resetwarnings()
            
            # Calculate performance metrics
            if self.baseline_success:
                speedup = self.baseline_time / solve_time
                iter_reduction = (self.baseline_iters - iters_gnp) / self.baseline_iters
            else:
                speedup = 0
                iter_reduction = 0
            
            # Performance score (higher is better)
            convergence_penalty = max(0, best_loss - 0.01)  # Penalty for poor convergence
            perf_score = speedup * (1 + iter_reduction) / (1 + convergence_penalty)
            
            # Additional bonuses for good performance
            if speedup > 1.0:
                perf_score *= 1.5  # Bonus for actual speedup
            if iter_reduction > 0.5:
                perf_score *= 1.2  # Bonus for significant iteration reduction
            if best_loss < 0.1:
                perf_score *= 1.3  # Bonus for good convergence
            
            # Record results
            result = {
                'config': config,
                'train_time': train_time,
                'train_loss_final': best_loss,
                'speedup': speedup,
                'iter_reduction': iter_reduction,
                'performance_score': perf_score,
                'gnp_iters': iters_gnp,
                'success': True
            }
            
            self.evaluation_history.append(result)
            
            if perf_score > self.best_score:
                self.best_score = perf_score
                self.best_config = config
                print(f"*** NEW BEST *** Score: {perf_score:.3f}, Speedup: {speedup:.2f}x, "
                      f"Iter reduction: {iter_reduction*100:.1f}%, Loss: {best_loss:.3e}")
            else:
                print(f"Score: {perf_score:.3f}, Speedup: {speedup:.2f}x, "
                      f"Iter reduction: {iter_reduction*100:.1f}%, Loss: {best_loss:.3e}")
            
            # Return negative score for minimization
            return -perf_score
            
        except Exception as e:
            print(f"Evaluation failed: {e}")
            return 10.0  # High penalty for failure


def run_bayesian_optimization(n_calls=30, n=16):
    """Run Bayesian optimization for GNP hyperparameters"""
    
    print(f"Starting Bayesian Optimization with {n_calls} evaluations")
    print("=" * 60)
    
    # Define search space
    search_space = [
        Categorical(['x_normal', 'x_subspace', 'x_mix'], name='training_data'),
        Integer(20, 100, name='m'),  # Krylov dimension
        Integer(16, 128, name='embed'),  # Embedding dimension
        Integer(3, 12, name='num_layers'),  # Number of layers
        Real(-4.0, -2.0, name='lr_log'),  # Log learning rate (1e-4 to 1e-2)
        Real(0.0, 0.3, name='drop_rate'),  # Dropout rate
    ]
    
    # Create objective function
    objective = GNPObjective(n=n, max_train_time=300)
    
    # Define objective with parameter names
    @use_named_args(search_space)
    def objective_func(**params):
        param_list = [params[dim.name] for dim in search_space]
        return objective(param_list)
    
    # Run Bayesian optimization
    print("Running Gaussian Process optimization...")
    result = gp_minimize(
        func=objective_func,
        dimensions=search_space,
        n_calls=n_calls,
        n_initial_points=10,  # Random exploration points
        acq_func='EI',  # Expected improvement
        random_state=42,
        verbose=True
    )
    
    # Extract best parameters
    best_params = {}
    for i, dim in enumerate(search_space):
        best_params[dim.name] = result.x[i]
    
    # Convert log learning rate back
    best_params['lr'] = 10 ** best_params['lr_log']
    del best_params['lr_log']
    
    print(f"\n{'='*60}")
    print("BAYESIAN OPTIMIZATION RESULTS")
    print(f"{'='*60}")
    print(f"Best objective value: {result.fun:.4f}")
    print(f"Best performance score: {-result.fun:.4f}")
    print(f"Best parameters: {best_params}")
    print(f"Number of evaluations: {len(result.func_vals)}")
    
    # Show optimization history
    print(f"\nOptimization history:")
    print(f"Initial best: {-result.func_vals[0]:.4f}")
    print(f"Final best: {-min(result.func_vals):.4f}")
    print(f"Improvement: {-min(result.func_vals) + result.func_vals[0]:.4f}")
    
    # Save results
    out_path = Path("./output")
    out_path.mkdir(exist_ok=True)
    
    # Save detailed results
    detailed_results = {
        'best_params': best_params,
        'best_score': -result.fun,
        'optimization_result': {
            'x': result.x,
            'fun': result.fun,
            'func_vals': result.func_vals.tolist(),
            'x_iters': [[float(val) if isinstance(val, (int, float)) else str(val) for val in x] for x in result.x_iters],
        },
        'evaluation_history': objective.evaluation_history
    }
    
    with open(out_path / 'bayesian_optimization_results.json', 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Convergence plot
    ax = axes[0, 0]
    plot_convergence(result, ax=ax)
    ax.set_title('Bayesian Optimization Convergence')
    
    # Performance over time
    ax = axes[0, 1]
    scores = [-val for val in result.func_vals]
    ax.plot(scores, 'b-', alpha=0.7)
    ax.plot(np.maximum.accumulate(scores), 'r-', linewidth=2, label='Best so far')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Performance Score')
    ax.set_title('Performance Score Evolution')
    ax.legend()
    ax.grid(True)
    
    # Parameter importance (simplified)
    ax = axes[1, 0]
    if len(objective.evaluation_history) > 5:
        # Simple correlation analysis
        param_names = ['training_data', 'm', 'embed', 'num_layers', 'lr', 'drop_rate']
        correlations = []
        
        for param in param_names:
            values = []
            scores = []
            for hist in objective.evaluation_history:
                if hist['success']:
                    if param == 'training_data':
                        # Convert categorical to numeric
                        val = {'x_normal': 0, 'x_subspace': 1, 'x_mix': 2}[hist['config'][param]]
                    else:
                        val = hist['config'][param]
                    values.append(val)
                    scores.append(hist['performance_score'])
            
            if len(values) > 2:
                corr = np.corrcoef(values, scores)[0, 1]
                correlations.append(abs(corr) if not np.isnan(corr) else 0)
            else:
                correlations.append(0)
        
        ax.barh(param_names, correlations)
        ax.set_xlabel('Absolute Correlation with Performance')
        ax.set_title('Parameter Importance')
    
    # Success rate analysis
    ax = axes[1, 1]
    if objective.evaluation_history:
        success_rate = len([h for h in objective.evaluation_history if h['success']]) / len(objective.evaluation_history)
        scores = [h['performance_score'] for h in objective.evaluation_history if h['success']]
        
        ax.hist(scores, bins=10, alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(scores), color='red', linestyle='--', label=f'Mean: {np.mean(scores):.3f}')
        ax.set_xlabel('Performance Score')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Score Distribution (Success rate: {success_rate:.1%})')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(out_path / 'bayesian_optimization_analysis.png', dpi=150, bbox_inches='tight')
    print(f"Analysis plots saved to {out_path / 'bayesian_optimization_analysis.png'}")
    
    return result, objective.best_config, objective.evaluation_history


def test_best_config(best_config, n=32):
    """Test the best configuration on a larger problem"""
    print(f"\n{'='*60}")
    print(f"Testing best config on larger problem (n={n})")
    print(f"Best config: {best_config}")
    print(f"{'='*60}")
    
    # Use the best config with some adaptations for larger problem
    config = best_config.copy()
    config['epochs'] = min(500, config['epochs'] + 200)  # More epochs for larger problem
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    A = gen_2d_poisson_matrix(n)
    b = build_rhs_2d(n)
    solver = GMRES()
    
    # Baseline
    print("Computing baseline...")
    start_time = time.time()
    x_baseline, iters_baseline, _, _, _ = solver.solve(
        A, b, M=None, restart=20, max_iters=3000, 
        timeout=120, rtol=1e-8, progress_bar=True)
    baseline_time = time.time() - start_time
    print(f"Baseline: {iters_baseline} iters, {baseline_time:.2f}s")
    
    # GNP
    print("Training and testing GNP...")
    net = ResGCN(A, config['num_layers'], config['embed'], 
                config['hidden'], config['drop_rate'], device, 
                scale_input=config['scale_input'], dtype=torch.float32).to(device)
    
    optimizer = torch.optim.AdamW(net.parameters(), lr=config['lr'], 
                                 weight_decay=config.get('weight_decay', 1e-5))
    
    M = GNP(A, config['training_data'], config['m'], net, device)
    
    train_start = time.time()
    hist_loss, best_loss, best_epoch, _ = M.train(
        config['batch_size'], 1, config['epochs'], optimizer, None,
        num_workers=0, progress_bar=True)
    train_time = time.time() - train_start
    
    solve_start = time.time()
    x_gnp, iters_gnp, _, _, _ = solver.solve(
        A, b, M=M, restart=20, max_iters=3000,
        timeout=120, rtol=1e-8, progress_bar=True)
    solve_time = time.time() - solve_start
    
    speedup = baseline_time / solve_time
    iter_reduction = (iters_baseline - iters_gnp) / iters_baseline
    
    print(f"\nResults on n={n}:")
    print(f"  Training time: {train_time:.1f}s")
    print(f"  Training loss: {hist_loss[0]:.3e} -> {best_loss:.3e}")
    print(f"  Solve time speedup: {speedup:.2f}x")
    print(f"  Iteration reduction: {iter_reduction*100:.1f}%")
    print(f"  Total time: {train_time + solve_time:.1f}s vs {baseline_time:.1f}s")
    
    return speedup > 1.0 and iter_reduction > 0


if __name__ == '__main__':
    # Run Bayesian optimization
    result, best_config, history = run_bayesian_optimization(n_calls=25, n=16)
    
    if best_config and len([h for h in history if h['success']]) > 0:
        print(f"\nFound working configuration!")
        
        # Test on larger problem
        success = test_best_config(best_config, n=32)
        
        if success:
            print(f"\n✓ Best config also works on larger problems!")
        else:
            print(f"\n⚠ Best config may not scale to larger problems")
    else:
        print(f"\nNo successful configurations found. Consider:")
        print(f"- Expanding search space")
        print(f"- Increasing number of evaluations")
        print(f"- Checking problem setup") 