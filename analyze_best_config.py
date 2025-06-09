import os
import time
import torch
import warnings
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from GNP.problems import *
from GNP.solver import GMRES
from GNP.preconditioners import *
from GNP.nn import ResGCN


def test_best_config():
    """Test the best configuration found by Bayesian optimization"""
    
    # Best configuration from Bayesian optimization
    best_config = {
        'training_data': 'x_normal',
        'm': 62,
        'embed': 61,
        'num_layers': 3,
        'lr': 0.008861577452533074,
        'drop_rate': 0.06983140212909128,
        'hidden': 122,  # embed * 2
        'epochs': 322,  # 200 + embed * 2
        'batch_size': 8,
        'weight_decay': 1e-5,
        'scale_input': True
    }
    
    print("=" * 60)
    print("TESTING BEST CONFIGURATION FROM BAYESIAN OPTIMIZATION")
    print("=" * 60)
    print(f"Best config: {best_config}")
    print(f"Performance score achieved: 6.0075")
    print(f"Expected: 1.52x speedup, 81.1% iteration reduction")
    print()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test on multiple problem sizes
    test_sizes = [16, 24, 32]
    results = []
    
    for n in test_sizes:
        print(f"\n{'='*40}")
        print(f"Testing on n = {n} (matrix size: {n*n}x{n*n})")
        print(f"{'='*40}")
        
        # Load problem
        A = gen_2d_poisson_matrix(n)
        b = build_rhs_2d(n)
        solver = GMRES()
        
        print(f'Matrix A size: {A.shape}')
        
        # Baseline solve
        print('\nComputing baseline (no preconditioner)...')
        start_time = time.time()
        x_baseline, iters_baseline, _, hist_rel_res_baseline, _ = solver.solve(
            A, b, M=None, restart=20, max_iters=3000, 
            timeout=120, rtol=1e-8, progress_bar=True)
        baseline_time = time.time() - start_time
        
        L2_error_baseline, _, _, _, _ = compute_L2_error(x_baseline, n)
        print(f'Baseline: {iters_baseline} iters, {baseline_time:.2f}s, L2 error: {L2_error_baseline:.2e}')
        
        # GNP with best config
        print('\nTraining GNP with best config...')
        
        # Adapt config for different problem sizes
        adapted_config = best_config.copy()
        if n > 16:
            # Scale up epochs for larger problems
            adapted_config['epochs'] = min(500, best_config['epochs'] + (n-16) * 10)
            # Slightly larger Krylov dimension
            adapted_config['m'] = min(100, best_config['m'] + (n-16) * 2)
        
        net = ResGCN(A, adapted_config['num_layers'], adapted_config['embed'], 
                    adapted_config['hidden'], adapted_config['drop_rate'], device, 
                    scale_input=adapted_config['scale_input'], dtype=torch.float32).to(device)
        
        optimizer = torch.optim.AdamW(net.parameters(), lr=adapted_config['lr'], 
                                     weight_decay=adapted_config['weight_decay'])
        
        M = GNP(A, adapted_config['training_data'], adapted_config['m'], net, device)
        
        train_start = time.time()
        hist_loss, best_loss, best_epoch, _ = M.train(
            adapted_config['batch_size'], 1, adapted_config['epochs'], optimizer, None,
            num_workers=0, progress_bar=True)
        train_time = time.time() - train_start
        
        print(f'Training: {train_time:.1f}s, loss: {hist_loss[0]:.3e} -> {best_loss:.3e} (epoch {best_epoch})')
        
        # GNP solve
        print('Solving with GNP preconditioner...')
        warnings.filterwarnings('error')
        try:
            solve_start = time.time()
            x_gnp, iters_gnp, _, hist_rel_res_gnp, _ = solver.solve(
                A, b, M=M, restart=20, max_iters=3000,
                timeout=120, rtol=1e-8, progress_bar=True)
            solve_time = time.time() - solve_start
            gnp_success = True
            
            L2_error_gnp, _, _, _, _ = compute_L2_error(x_gnp, n)
            
        except Exception as e:
            print(f'GNP solve failed: {e}')
            gnp_success = False
            solve_time = np.inf
            iters_gnp = 3000
            L2_error_gnp = np.inf
            hist_rel_res_gnp = [1.0]
        
        warnings.resetwarnings()
        
        if gnp_success:
            speedup = baseline_time / solve_time
            iter_reduction = (iters_baseline - iters_gnp) / iters_baseline
            total_time = train_time + solve_time
            
            print(f'\nResults for n={n}:')
            print(f'  GNP: {iters_gnp} iters, {solve_time:.2f}s solve, L2 error: {L2_error_gnp:.2e}')
            print(f'  Speedup: {speedup:.2f}x')
            print(f'  Iteration reduction: {iter_reduction*100:.1f}%')
            print(f'  Total time (train + solve): {total_time:.1f}s vs baseline {baseline_time:.1f}s')
            print(f'  Training loss quality: {best_loss:.3e}')
            
            # Check if GNP is actually beneficial
            beneficial = speedup > 1.0 and iter_reduction > 0.3
            print(f'  ✓ Beneficial: {beneficial}')
            
            results.append({
                'n': n,
                'baseline_time': baseline_time,
                'baseline_iters': iters_baseline,
                'train_time': train_time,
                'solve_time': solve_time,
                'total_time': total_time,
                'gnp_iters': iters_gnp,
                'speedup': speedup,
                'iter_reduction': iter_reduction,
                'train_loss': best_loss,
                'L2_error_baseline': L2_error_baseline,
                'L2_error_gnp': L2_error_gnp,
                'beneficial': beneficial,
                'success': True
            })
        else:
            print(f'\n✗ GNP failed for n={n}')
            results.append({
                'n': n,
                'success': False
            })
    
    # Create summary analysis
    print(f"\n{'='*60}")
    print("SUMMARY ANALYSIS")
    print(f"{'='*60}")
    
    successful_results = [r for r in results if r.get('success', False)]
    
    if successful_results:
        print(f"Successful tests: {len(successful_results)}/{len(results)}")
        print()
        print(f"{'Size':<6} {'Speedup':<8} {'Iter Red':<10} {'Train Loss':<12} {'Beneficial'}")
        print("-" * 60)
        
        for r in successful_results:
            print(f"{r['n']:<6} {r['speedup']:<8.2f} {r['iter_reduction']*100:<10.1f}% "
                  f"{r['train_loss']:<12.3e} {'✓' if r['beneficial'] else '✗'}")
        
        # Overall assessment
        beneficial_count = sum(1 for r in successful_results if r['beneficial'])
        success_rate = beneficial_count / len(successful_results)
        
        print(f"\nOverall success rate: {success_rate:.1%}")
        
        if success_rate > 0.5:
            print("✓ Best configuration is EFFECTIVE!")
            print("\nKey insights from Bayesian optimization:")
            print("- training_data='x_normal' is crucial (much better than x_subspace)")
            print("- Shallow networks (3 layers) work better than deep ones")
            print("- Higher learning rate (~0.009) is needed")
            print("- Medium embedding dimension (61) is optimal")
            print("- Light dropout (0.07) helps")
        else:
            print("⚠ Configuration needs further tuning")
    
    # Create plots
    if len(successful_results) > 1:
        create_analysis_plots(successful_results, best_config)
    
    return successful_results


def create_analysis_plots(results, config):
    """Create analysis plots"""
    out_path = Path("./output")
    out_path.mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Extract data
    sizes = [r['n'] for r in results]
    speedups = [r['speedup'] for r in results]
    iter_reductions = [r['iter_reduction'] * 100 for r in results]
    train_losses = [r['train_loss'] for r in results]
    
    # Speedup vs problem size
    ax = axes[0, 0]
    ax.semilogx(sizes, speedups, 'o-', color='green', linewidth=2, markersize=8)
    ax.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='No speedup')
    ax.set_xlabel('Problem size n')
    ax.set_ylabel('Speedup factor')
    ax.set_title('GNP Speedup vs Problem Size')
    ax.grid(True)
    ax.legend()
    
    # Iteration reduction vs problem size
    ax = axes[0, 1]
    ax.semilogx(sizes, iter_reductions, 'o-', color='blue', linewidth=2, markersize=8)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='No reduction')
    ax.set_xlabel('Problem size n')
    ax.set_ylabel('Iteration reduction (%)')
    ax.set_title('Iteration Reduction vs Problem Size')
    ax.grid(True)
    ax.legend()
    
    # Training loss vs problem size
    ax = axes[1, 0]
    ax.semilogy(sizes, train_losses, 'o-', color='orange', linewidth=2, markersize=8)
    ax.set_xlabel('Problem size n')
    ax.set_ylabel('Training loss')
    ax.set_title('Training Loss vs Problem Size')
    ax.grid(True)
    
    # Configuration summary
    ax = axes[1, 1]
    ax.axis('off')
    config_text = "Best Configuration:\n\n"
    for key, value in config.items():
        if key in ['training_data', 'num_layers', 'lr', 'embed', 'm', 'drop_rate']:
            if isinstance(value, float):
                config_text += f"{key}: {value:.4f}\n"
            else:
                config_text += f"{key}: {value}\n"
    
    ax.text(0.1, 0.9, config_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(out_path / 'best_config_analysis.png', dpi=150, bbox_inches='tight')
    print(f"Analysis plots saved to {out_path / 'best_config_analysis.png'}")


if __name__ == '__main__':
    results = test_best_config() 