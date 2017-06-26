import numpy as np
import matplotlib.pyplot as plt
from assembly import assembly
from solver import solver


def preprocess():
    # Get element length in each region
    le = np.array(
        (d1 / num_elements_in_region[0], d2 / num_elements_in_region[1]))
    return le


def fem(le, beta_values, num_elements_in_region, total_num_elements, EpsilonR,
        MuR):
    # Allocate space for eigenvalues
    allEig = np.zeros((total_num_elements + 1, np.size(beta_values)),
                      dtype='complex128')

    # Sweep beta parameter
    for i in range(0, np.size(beta_values)):
        A, B = assembly(num_elements_in_region, total_num_elements, le,
                        beta_values[i], EpsilonR, MuR)
        w = solver(A, B)
        allEig[:, i] = w
    return allEig


def postprocess(allEig, beta_values, d):
    # Calculate k0
    k0 = np.sqrt(allEig)

    # Plot k0 vs beta*d
    for i in range(200):
        plt.plot(beta_values*d, k0[i, :], '.')
    plt.ylim(0, 25)
    plt.show()
    return None


if __name__ == '__main__':
    # Structure parameters
    MuR = 1
    EpsilonR = np.array((9, 1))

    d1 = 100e-3
    d2 = d1
    d = d1 + d2

    # Simulation parameters
    num_elements_in_region = np.array((100, 100))
    total_num_elements = num_elements_in_region[0] + num_elements_in_region[1]

    # Source parameter
    beta_values = np.linspace(0, np.pi / d, 50)

    # Pre-process step
    le = preprocess()
    # FEM step
    allEig = fem(le, beta_values, num_elements_in_region, total_num_elements,
                 EpsilonR, MuR)
    # Post-process step
    postprocess(allEig, beta_values, d)
