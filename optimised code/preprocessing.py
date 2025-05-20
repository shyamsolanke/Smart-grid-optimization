import cupy as cp

def build_lagged_matrix_gpu(y, exog, p, b):
    """
    GPU-optimized builder for the ARX design matrix and target vector.
    
    Parameters:
    - y: array-like of shape (T,), target series (can be NumPy or CuPy)
    - exog: array-like of shape (T, M), exogenous features
    - p: int, number of autoregressive lags
    - b: int, number of exogenous lags
    
    Returns:
    - A_gpu: CuPy array of shape (T-L, p + M*b), design matrix on GPU
    - y_target_gpu: CuPy array of shape (T-L,), aligned targets
    """
    # Move data to GPU
    y_gpu = cp.asarray(y, dtype=cp.float64)
    exog_gpu = cp.asarray(exog, dtype=cp.float64)
    
    T = y_gpu.shape[0]
    M = exog_gpu.shape[1]
    L = max(p, b)
    N = T - L
    
    # Initialize design matrix and target
    A_gpu = cp.zeros((N, p + M * b), dtype=cp.float64)
    y_target_gpu = y_gpu[L:]
    
    # Vectorized filling of AR part (lags of y)
    for k in range(1, p+1):
        A_gpu[:, k-1] = y_gpu[L-k : T-k]
    
    # Vectorized filling of exogenous part
    for j in range(1, b+1):
        start = p + (j-1) * M
        A_gpu[:, start:start+M] = exog_gpu[L-j : T-j, :]
    
    return A_gpu, y_target_gpu


