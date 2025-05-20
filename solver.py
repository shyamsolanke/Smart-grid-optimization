import math

def transpose(matrix):
    # Transpose of a matrix
    return [list(row) for row in zip(*matrix)]

def matmul(A, B):
    # Multiply two matrices A (m×n) and B (n×p)
    return [
        [
            sum(A[i][k] * B[k][j] for k in range(len(B)))
            for j in range(len(B[0]))
        ]
        for i in range(len(A))
    ]

def matvec(A, v):
    # Multiply matrix A (m×n) with vector v (n,)
    return [sum(A[i][j] * v[j] for j in range(len(v))) for i in range(len(A))]

def identity(n):
    # Identity matrix of size n×n
    I = [[0.0]*n for _ in range(n)]
    for i in range(n):
        I[i][i] = 1.0
    return I

def augment_matrix(A, B):
    # Augment matrix A (n×n) with B (n×m) side-by-side
    return [A[i] + B[i] for i in range(len(A))]

def gauss_jordan(A):
    # Perform Gauss-Jordan elimination on augmented matrix A (n×2n)
    n = len(A)
    m = len(A[0])
    for i in range(n):
        # Pivot normalization
        pivot = A[i][i]
        if abs(pivot) < 1e-12:
            # Swap with a non-zero pivot row
            for k in range(i+1, n):
                if abs(A[k][i]) > 1e-12:
                    A[i], A[k] = A[k], A[i]
                    pivot = A[i][i]
                    break
            else:
                raise ValueError("Matrix is singular and cannot be inverted")
        # Scale row to make pivot = 1
        for j in range(m):
            A[i][j] /= pivot
        # Eliminate other rows
        for k in range(n):
            if k != i:
                factor = A[k][i]
                for j in range(m):
                    A[k][j] -= factor * A[i][j]
    return A

def invert_matrix(A):
    # Invert a square matrix A (n×n) using Gauss-Jordan
    n = len(A)
    # Create augmented [A | I]
    augmented = augment_matrix([row[:] for row in A], identity(n))
    # Apply Gauss-Jordan to get [I | A⁻¹]
    result = gauss_jordan(augmented)
    # Extract and return the right half
    return [row[n:] for row in result]

def solve_theta(A, y):
    """
    Solve θ = (A^T A)^(-1) A^T y without external libraries.
    A: list of lists (design matrix m×k)
    y: list (target vector m)
    Returns θ as a list (k)
    """
    At = transpose(A)                # k×m
    AtA = matmul(At, A)             # k×k
    Aty = matvec(At, y)             # k
    AtA_inv = invert_matrix(AtA)    # k×k
    theta = matvec(AtA_inv, Aty)    # k
    return theta

# # Example usage with small numeric data:
# A_example = [
#     [1, 2],
#     [3, 4],
#     [5, 6]
# ]
# y_example = [7, 8, 9]
# theta_example = solve_theta(A_example, y_example)
# print("Estimated θ:", theta_example)
