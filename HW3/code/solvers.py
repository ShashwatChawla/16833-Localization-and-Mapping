'''
    Initially written by Ming Hsiao in MATLAB
    Rewritten in Python by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

from scipy.sparse import csc_matrix, eye
from scipy.sparse.linalg import inv, splu, spsolve, spsolve_triangular
from sparseqr import rz, permutation_vector_to_matrix, solve as qrsolve
import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse import csr_matrix



def solve_default(A, b):
    from scipy.sparse.linalg import spsolve
    x = spsolve(A.T @ A, A.T @ b)
    return x, None


def solve_pinv(A, b):
    # TODO: return x s.t. Ax = b using pseudo inverse.
    
    # Formula x = (A^T A)^-1 A^T b
    pseudo_inverse = inv(A.T @ A) 
    x = pseudo_inverse @ A.T @ b
    return x, None


# Forward substitution to solve L * y = b
def forward_substitution(L, b):
    y = np.zeros_like(b)
    
    for i in range(len(b)):
        y[i] = b[i] - np.dot(L[i, :i].todense(), y[:i]) 
    return y

# Backward substitution to solve U * x = y
def backward_substitution(U, y):
    
    x = np.zeros_like(y)
    
    for i in range(len(y)-1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i+1:].todense(), x[i+1:])) / U[i, i]
    
    return x

def solve_lu(A, b):
    # TODO: return x, U s.t. Ax = b, and A = LU with LU decomposition.
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.splu.html
    
    lu = splu(A.T @ A, permc_spec='NATURAL') 
    
    # Use this internal function for faster convergence
    # x = lu.solve(A.T @ b)
    
        
    # Forward substitution
    y = forward_substitution(lu.L, A.T @ b)
    
    # Backward substitution
    x = backward_substitution(lu.U, y)

    return x, lu.U


def solve_lu_colamd(A, b):
    # TODO: return x, U s.t. Ax = b, and Permutation_rows A Permutation_cols = LU with reordered LU decomposition.
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.splu.html
    
    lu = splu(A.T @ A, permc_spec='COLAMD') 
    # Backward Substitution U x = y & Forward Substution L y = A^T B 
    # TODO@Shashwat: Validate if solve() performs both
    x = lu.solve(A.T @ b)
    
    return x, lu.U


def solve_qr(A, b):
    # TODO: return x, R s.t. Ax = b, and |Ax - b|^2 = |Rx - d|^2 + |e|^2
    # https://github.com/theNded/PySPQR
    
    z, R, E, rank = rz(A, b, permc_spec='NATURAL')
    # R is upper triangular (CSR format)
    x = spsolve_triangular(csr_matrix(R), z.flatten(), lower=False)
    return x, R


def solve_qr_colamd(A, b):
    # TODO: return x, R s.t. Ax = b, and |Ax - b|^2 = |R E^T x - d|^2 + |e|^2, with reordered QR decomposition (E is the permutation matrix).
    # https://github.com/theNded/PySPQR
    
    z, R, E, rank = rz(A, b, permc_spec='COLAMD')
    
    E = permutation_vector_to_matrix(E)
    # R is upper triangular (CSR format)
    x = spsolve_triangular(csr_matrix(R), z.flatten(), lower=False)
    
    # Reorder X using E(Permutation matrix)
    x = E @ x 
    return x, R


def solve(A, b, method='default'):
    '''
    \param A (M, N) Jacobian matrix
    \param b (M, 1) residual vector
    \return x (N, 1) state vector obtained by solving Ax = b.
    '''
    M, N = A.shape

    fn_map = {
        'default': solve_default,
        'pinv': solve_pinv,
        'lu': solve_lu,
        'qr': solve_qr,
        'lu_colamd': solve_lu_colamd,
        'qr_colamd': solve_qr_colamd,
    }

    return fn_map[method](A, b)
