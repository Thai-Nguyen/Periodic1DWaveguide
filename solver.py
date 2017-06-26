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
    w, v = sp.linalg.eig(A, B)

    # Convert to column vectors
    # w = scipy.transpose(w)
    # v = scipy.transpose(v)

    # Sort eigenvalues
    w = scipy.sort(w)

    return w
