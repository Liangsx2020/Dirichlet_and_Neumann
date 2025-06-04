import os
import time
import torch
import warnings
from pathlib import Path
import matplotlib.pyplot as plt

from GNP.problems import *
from GNP.solver import GMRES
from GNP.preconditioners import *
from GNP.nn import ResGCN


#-----------------------------------------------------------------------------
def main():

    # Setup and parameters

    n = 8


    restart = 10                # restart cycle in GMRES
    max_iters = 10000           # maximum number of GMRES iterations
    timeout = None              # timeout in seconds
    rtol = 1e-12                # relative residual tolerance in GMRES
    training_data = 'x_mix'     # type of training data x
    m = 40                      # Krylov subspace dimension for training data
    num_layers = 8              # number of layers in GNP
    embed = 16                  # embedding dimension in GNP
    hidden = 32                 # hidden dimension in MLPs in GNP
    drop_rate = 0.0             # dropout rate in GNP
    disable_scale_input = False # whether disable the scaling of inputs in GNP
    dtype = torch.float32       # training precision for GNP
    batch_size = 16             # batch size in training GNP
    grad_accu_steps = 1         # gradient accumulation steps in training GNP
    epochs = 2000               # number of epochs in training GNP
    lr = 1e-3                   # learning rate in training GNP
    weight_decay = 0.0          # weight decay in training GNP
    save_model = True           # whether save model
    hide_solver_bar = False     # whether hide progress bar in linear solver
    hide_training_bar = False   # whether hide progress bar in GNP training
    num_workers = 4             # number of dataloader workers in training GNP

    # computing device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load problem
    A = gen_2d_poisson_matrix(n)

    # Right-hand side b
    b = build_rhs_2d(n)

    # Output path and filename
    out_path = os.path.abspath(os.path.expanduser("./output"))
    Path(out_path).mkdir(parents=True, exist_ok=True)
    out_file_prefix = f"poisson_{n}_"
    out_file_prefix_with_path = os.path.join(out_path, out_file_prefix)

    # Solver
    solver = GMRES()

    print(f'Matrix A size: {A.shape}')
    print(f'Right-hand side size: {b.shape}')

    # GMRES without preconditioner
    print('\nSolving linear system using GMRES without preconditioner ...')


    start_time = time.time()
    x_gmres, _, _, hist_rel_res, hist_time = solver.solve(
        A, b, M=None, restart=restart, max_iters=max_iters,
        timeout=timeout, rtol=rtol, progress_bar=not hide_solver_bar)
    end_time = time.time()

    solving_time = end_time - start_time
    print(f'Done. Final relative residual = {hist_rel_res[-1]:.4e}')
    print(f'{device} time: {solving_time:.4f} seconds')

    # exact solution
    # p_exact, p_exact_grid, X, Y = generate_exact_solution(n)

    # L2 norm
    L2_error, _, _, _, _ = compute_L2_error(x_gmres, n)
    print(f'L2 error without preconditioner: {L2_error:.4e}')


    

    # GMRES with GNP: Train preconditioner
    print('\nTraining GNP ...')
    net = ResGCN(A, num_layers, embed, hidden, drop_rate,device,
                 scale_input=disable_scale_input, dtype=dtype).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr,
                                 weight_decay=weight_decay)
    scheduler = None
    M = GNP(A, training_data, m, net, device)
    tic = time.time()
    hist_loss, best_loss, best_epoch, model_file = M.train(
        batch_size, grad_accu_steps, epochs, optimizer, scheduler,
        num_workers=num_workers,
        checkpoint_prefix_with_path=\
        out_file_prefix_with_path if save_model else None,
        progress_bar=not hide_training_bar)
    print(f'Done. Training time: {time.time() - tic:.4f} seconds')
    print(f'Loss: inital = {hist_loss[0]}, '
          f'final = {hist_loss[-1]}, '
          f'best = {best_loss}, epoch = {best_epoch}')
    if save_model:
        print(f'Best model saved in {model_file}')
    
    # Investigate training history of the preconditioner
    print('\nPlotting training history ...')
    plt.figure(1)
    plt.semilogy(hist_loss, label='train')
    plt.title(f'poisson_{n}: Preconditioner convergence (MAE loss)')
    plt.legend()
    # plt.show()
    full_path = out_file_prefix_with_path + 'training.png'
    plt.savefig(full_path)
    print(f'Figure saved in {full_path}')
    
    # Load the best checkpoint
    if model_file:
        print(f'\nLoading model from {model_file} ...')
        net.load_state_dict(torch.load(model_file, map_location=device))
        M = GNP(A, training_data, m, net, device)
        print('Done.')
    else:
        print('\nNo checkpoint is saved. Use model from the last epoch.')
    
    # GMRES with GNP: Linear solve
    print('\nSolving linear system using GMRES with GNP ...')
    warnings.filterwarnings('error')
    try:
        start_time_gnp = time.time()
        x_gmres_gnp, _, _, hist_rel_res_gnp, hist_time_gnp = solver.solve(
            A, b, M=M, restart=restart, max_iters=max_iters,
            timeout=timeout, rtol=rtol, progress_bar=not hide_solver_bar)
        end_time_gnp = time.time()
        solving_time_gnp = end_time_gnp - start_time_gnp
    
    except UserWarning as w:
        print('Warning:', w)
        print('GMRES preconditioned by GNP fails')
        hist_rel_res_gnp = None
        hist_time_gnp = None
    else:
        print(f'Done. Final relative residual = {hist_rel_res_gnp[-1]:.4e}')
        print(f'{device} time: {solving_time_gnp:.4f} seconds')
        L2_error_gnp, _, _, _, _ = compute_L2_error(x_gmres_gnp, n)
        print(f'L2 error of GNP: {L2_error_gnp:.4e}')
    warnings.resetwarnings()
    
    # Investigate solution history
    print('\nPlotting solution history ...')
    plt.figure(2)
    plt.semilogy(hist_rel_res, color='C0', label='no precond')
    if hist_rel_res_gnp is not None:
        plt.semilogy(hist_rel_res_gnp, color='C7', label='GNP')
    solver_name = solver.__class__.__name__
    plt.title(f'poisson_{n}: {solver_name} convergence (relative residual)')
    plt.xlabel('(Outer) Iterations')
    plt.legend()
    # plt.show()
    full_path = out_file_prefix_with_path + 'solver.png'
    plt.savefig(full_path)
    print(f'Figure saved in {full_path}')
    
    # Compare solution speed
    print('\nPlotting solution history (time to solution) ...')
    plt.figure(3)
    plt.semilogy(hist_time, hist_rel_res, color='C0', label='no precond')
    if hist_rel_res_gnp is not None:
        plt.semilogy(hist_time_gnp, hist_rel_res_gnp, color='C7', label='GNP')
    solver_name = solver.__class__.__name__
    plt.title(f'poisson_{n}: {solver_name} convergence (relative residual)')
    plt.xlabel('Time (seconds)')
    plt.legend()
    # plt.show()
    full_path = out_file_prefix_with_path + 'time.png'
    plt.savefig(full_path)
    print(f'Figure saved in {full_path}')
    

#-----------------------------------------------------------------------------
if __name__ == '__main__':
    main()