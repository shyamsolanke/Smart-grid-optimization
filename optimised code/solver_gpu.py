import cupy as cp

# GPU-accelerated solver module (`solver_gpu.py`)

def solve_theta_gpu(A, y):
    """
    GPU-optimized closed-form solver for θ = (A^T A)^(-1) A^T y using CuPy.
    A: array-like or CuPy array of shape (m, k)
    y: array-like or CuPy array of shape (m,)
    Returns θ as a NumPy array of shape (k,)
    """
    # Transfer to GPU (if not already)
    A_gpu = cp.asarray(A, dtype=cp.float64)   # cast to double precision
    y_gpu = cp.asarray(y, dtype=cp.float64)
    
    # Compute A^T A and A^T y on GPU
    AtA_gpu = A_gpu.T @ A_gpu                 # (k x k) matrix
    Aty_gpu = A_gpu.T @ y_gpu                 # (k,) vector
    
    # Solve the linear system on GPU
    theta_gpu = cp.linalg.solve(AtA_gpu, Aty_gpu)
    
    # Move result back to CPU as a NumPy array
    return theta_gpu

# Example usage:
if __name__ == "__main__":
    # sample data
    A = [[1, 2], [3, 4], [5, 6]]
    y = [7, 8, 9]
    
    theta = solve_theta_gpu(A, y)
    print("Estimated θ (GPU):", theta)
