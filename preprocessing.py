import numpy as np

# 3) Build the design matrix for those fixed orders
def build_lagged_matrix(y, exog, p, b):
    T = len(y)
    M = exog.shape[1]
    L = max(p, b)
    N = T - L
    A = np.zeros((N, p + M * b))
    y_target = y[L:]
    for i in range(N):
        t = i + L
        A[i, :p] = y[t-p:t][::-1]               # AR part
        for k in range(1, b+1):                 # exogenous part
            start = p + (k-1)*M
            A[i, start:start+M] = exog[t-k]
    return A, y_target

