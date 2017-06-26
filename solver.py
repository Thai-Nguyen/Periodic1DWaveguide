import scipy as sp
import scipy.linalg


def solver(A, B):
    '''
    Solves the eigenvalue problem Ax = kBx where k is the eigenvalue
    :param A:
    :param B:
    :return:
    '''
    # Get eigenvalues and corresponding eigenvectors
    eig_values, eig_vectors = sp.linalg.eig(A, B)

    # Sort eigenvalues
    id = eig_values.argsort()[::1]
    eig_values = eig_values[id]
    eig_vectors = eig_vectors[:, id]

    return eig_values, eig_vectors
