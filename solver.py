import scipy as sp
import scipy.linalg


def solver(A, B):
    # Get eigenvalues and corresponding eigenvectors
    w, v = sp.linalg.eig(A, B)
    return w
