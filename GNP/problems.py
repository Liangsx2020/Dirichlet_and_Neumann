import torch
import numpy as np


def gen_2d_poisson_matrix(N):
    """
    Create a discrete matrix for 2D Laplace operator using 5-point stencil with mixed boundary conditions

    Boundary conditions:
    - Top (j = N-1): Dirichlet, p = 0 (including corners)
    - Bottom (j = 0): Neumann, ∂p/∂y = 0
    - Left (i = 0): Neumann, ∂p/∂x = 0
    - Right (i = N-1): Neumann, ∂p/∂x = 0

    For Neumann boundaries: set ghost points to zero, reduce center coefficient by 1

    Grid indexing convention:
    - i: x direction, from left to right (i = 0, 1, ..., N-1)
    - j: y direction, from bottom to top (j = 0, 1, ..., N-1)
    - Linear index: idx = i * N + j

    Parameters:
        N: Number of grid points in each dimension
        domain_size: Size of computational domain (default is unit square)
    Returns:
        A: Discretized Laplace operator in sparse matrix form (torch.sparse_csc_tensor)
    """
    total_size = N * N
    row_indices = []
    col_indices = []
    values = []

    for j in range(N):  # y方向：从下到上 (j=0 底部, j=N-1 顶部)
        for i in range(N):  # x方向：从左到右 (i=0 左侧, i=N-1 右侧)
            idx = j * N + i  # 新的索引计算：行主序（row-major）

            # Top boundary (j = N-1): Dirichlet
            if j == N - 1:
                row_indices.append(idx)
                col_indices.append(idx)
                values.append(1.0)

            # Bottom boundary (j = 0): Neumann
            elif j == 0:
                if i == 0:  # Bottom-left corner
                    row_indices.append(idx)
                    col_indices.append(idx)
                    values.append(2.0)  # 4 - 2 = 2

                    # Right neighbor (i+1, j)
                    row_indices.append(idx)
                    col_indices.append(j * N + (i + 1))
                    values.append(-1.0)

                    # Upper neighbor (i, j+1)
                    row_indices.append(idx)
                    col_indices.append((j + 1) * N + i)
                    values.append(-1.0)

                elif i == N - 1:  # Bottom-right corner
                    row_indices.append(idx)
                    col_indices.append(idx)
                    values.append(2.0)

                    # Left neighbor (i-1, j)
                    row_indices.append(idx)
                    col_indices.append(j * N + (i - 1))
                    values.append(-1.0)

                    # Upper neighbor (i, j+1)
                    row_indices.append(idx)
                    col_indices.append((j + 1) * N + i)
                    values.append(-1.0)

                else:  # Bottom edge (interior)
                    row_indices.append(idx)
                    col_indices.append(idx)
                    values.append(3.0)  # 4 - 1 = 3

                    # Left neighbor
                    row_indices.append(idx)
                    col_indices.append(j * N + (i - 1))
                    values.append(-1.0)

                    # Right neighbor
                    row_indices.append(idx)
                    col_indices.append(j * N + (i + 1))
                    values.append(-1.0)

                    # Upper neighbor
                    row_indices.append(idx)
                    col_indices.append((j + 1) * N + i)
                    values.append(-1.0)

            # Left boundary (i = 0, j != 0, j != N-1): Neumann
            elif i == 0:
                row_indices.append(idx)
                col_indices.append(idx)
                values.append(3.0)  # 4 - 1 = 3

                # Right neighbor
                row_indices.append(idx)
                col_indices.append(j * N + (i + 1))
                values.append(-1.0)

                # Upper neighbor
                row_indices.append(idx)
                col_indices.append((j + 1) * N + i)
                values.append(-1.0)

                # Lower neighbor
                row_indices.append(idx)
                col_indices.append((j - 1) * N + i)
                values.append(-1.0)

            # Right boundary (i = N-1, j != 0, j != N-1): Neumann
            elif i == N - 1:
                row_indices.append(idx)
                col_indices.append(idx)
                values.append(3.0)  # 4 - 1 = 3

                # Left neighbor
                row_indices.append(idx)
                col_indices.append(j * N + (i - 1))
                values.append(-1.0)

                # Upper neighbor
                row_indices.append(idx)
                col_indices.append((j + 1) * N + i)
                values.append(-1.0)

                # Lower neighbor
                row_indices.append(idx)
                col_indices.append((j - 1) * N + i)
                values.append(-1.0)

            # Interior points: standard 5-point stencil
            else:
                row_indices.append(idx)
                col_indices.append(idx)
                values.append(4.0)

                # Left neighbor
                row_indices.append(idx)
                col_indices.append(j * N + (i - 1))
                values.append(-1.0)

                # Right neighbor
                row_indices.append(idx)
                col_indices.append(j * N + (i + 1))
                values.append(-1.0)

                # Lower neighbor
                row_indices.append(idx)
                col_indices.append((j - 1) * N + i)
                values.append(-1.0)

                # Upper neighbor
                row_indices.append(idx)
                col_indices.append((j + 1) * N + i)
                values.append(-1.0)

    # 转换为PyTorch稀疏矩阵 (保持不变)
    row_indices = torch.tensor(row_indices, dtype=torch.long)
    col_indices = torch.tensor(col_indices, dtype=torch.long)
    values = torch.tensor(values, dtype=torch.float64)

    A = torch.sparse_coo_tensor(
        torch.stack([row_indices, col_indices]),
        values,
        (total_size, total_size)
    ).coalesce().to_sparse_csc()

    return A


def build_rhs_2d(N, domain_size=1.0):
    h = domain_size / (N - 1)
    
    # Initialize right-hand side vector
    b = torch.zeros(N * N, dtype=torch.float64)
    
    # Use same indexing as matrix generation function: row-major order
    for j in range(N):  # y direction: bottom to top (j=0 bottom, j=N-1 top)
        for i in range(N):  # x direction: left to right (i=0 left, i=N-1 right)
            x = i * h
            y = j * h
            
            # Calculate index: idx = j * N + i (row-major, consistent with matrix function)
            idx = j * N + i
            
            # Calculate second derivatives
            d2p_dx2 = 2 * (1 - 6 * x + 6 * x ** 2) * y ** 2 * (1 - y) ** 2
            d2p_dy2 = 2 * (1 - 6 * y + 6 * y ** 2) * x ** 2 * (1 - x) ** 2
            f_val = -(d2p_dx2 + d2p_dy2) * (h ** 2)
            
            b[idx] = f_val
    
    # Handle top boundary (Dirichlet condition): all points where j = N-1
    for i in range(N):
        idx = (N - 1) * N + i  # Top boundary index (row-major)
        b[idx] = 0.0
    
    return b


def generate_exact_solution(N, domain_size=1.0):
    """
    Generate the exact solution of the Poisson equation p(x,y) = x²y²(1-x)²(1-y)²

    Parameters:
    N : int
        Number of grid points in each direction (including boundary points)
    domain_size : float
        Size of computational domain (default is unit square)

    Returns:
    p_exact : torch.Tensor
        Exact solution vector of size (N*N)
    p_exact_grid : torch.Tensor
        Exact solution reconstructed on 2D grid of size (N, N)
    X, Y : torch.Tensor
        x and y coordinates of grid points
    """
    # Generate grid point coordinates
    x = torch.linspace(0, domain_size, N, dtype=torch.float64) # 创建均匀分布的网格点
    y = torch.linspace(0, domain_size, N, dtype=torch.float64) # 创建均匀分布的网格点
    X, Y = torch.meshgrid(x, y, indexing='ij') # 生成二维网格

    # Calculate exact solution p = x²y²(1-x)²(1-y)²
    p_exact_grid = X ** 2 * Y ** 2 * (1 - X) ** 2 * (1 - Y) ** 2
    p_exact = p_exact_grid.flatten()

    return p_exact, p_exact_grid, X, Y


def compute_L2_error(numerical_solution, N, domain_size=1.0):
    """
    Compute the L2 error between the numerical solution and the exact solution.

    Parameters:
        numerical_solution: numerical solution vector with size N * N
        N: number of grid points on each boundary
        domain_size: size of the computational domain

    Returns:
        L2_error: float
            L2 error of the numerical solution
        p_exact_grid: torch.Tensor
            Exact solution reconstructed onto a two-dimensional grid
        p_h_grid: torch.Tensor
            Numerical solution reconstructed onto a two-dimensional grid
        X, Y: torch.Tensor
            Grid points' x and y coordinates
    """
    # Get exact solution
    p_exact, p_exact_grid, X, Y = generate_exact_solution(N, domain_size)

    # Restructure the numerical solution into a two-dimensional grid
    p_h_grid = numerical_solution.reshape(N, N)

    # Compute the pointwise error
    error = p_exact_grid - p_h_grid

    # Compute the L2 norm of the error with proper scaling
    h = domain_size / (N - 1)
    L2_error = torch.sqrt(torch.sum(error ** 2) * h ** 2)

    return L2_error, p_exact_grid, p_h_grid, X, Y
    

if __name__ == '__main__':
    # Test the function build_rhs_2d
    # N = 4
    # b = build_rhs_2d(N, domain_size=1.0)
    # print(b)

    # Test the function gen_2d_poisson_matrix
    N = 3
    A = gen_2d_poisson_matrix(N)
    b = build_rhs_2d(N, domain_size=1.0)
    print(A)
    print(b)