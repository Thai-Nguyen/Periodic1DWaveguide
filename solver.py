import scipy as sp
import scipy.linalg


def solver(A, B):
    # Get eigenvalues and eigenvectors (ignore left eigenvector)
    w, v = sp.linalg.eig(A, B)
    return w
