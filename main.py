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

    A, B = assembly(num_elements_in_region, total_num_elements, le,
                        beta_values, EpsilonR, MuR)
    eig_values, eig_vectors = solver(A, B)
    return eig_values, eig_vectors


def plot_dispersion(allEig, beta_values, d):
    # Calculate k0
    k0 = np.sqrt(allEig)

    # Plot k0 vs beta*d
    num_eig, _ = allEig.shape
    for i in range(num_eig):
        plt.plot(beta_values * d, k0[i, :], '.')
    plt.grid()
    plt.xlim(0, np.pi)
    plt.ylim(0, 25)
    plt.xlabel('$\\beta d$')
    plt.ylabel('$k_0$')
    plt.show()
    return None


def plot_electric_field(eig_value, eig_vector, beta, d):
    # Plot the first three modes
    for i in range(3):
        print('i', i, 'k0', np.sqrt(eig_value[i]))
        plt.plot(eig_vector[:, i])
    plt.grid()
    plt.show()
    return None


def run(beta, d, num_elements_in_region, total_num_elements, EpsilonR, MuR):
    le = preprocess()
    eig_value, eig_vector = fem(le, beta, num_elements_in_region, total_num_elements,
        EpsilonR, MuR)
    plot_electric_field(eig_value, eig_vector, beta, d)
    return None


def sweep_parameter(beta_values, d, num_elements_in_region, total_num_elements, EpsilonR, MuR):
    le = preprocess()
    # Allocate space for eigenvalues
    all_eig = np.zeros((total_num_elements, np.size(beta_values)), dtype='complex128')
    for i in range(np.size(beta_values)):
        all_eig[:, i], _ = fem(le, beta_values[i], num_elements_in_region, total_num_elements, EpsilonR, MuR)
    plot_dispersion(all_eig, beta_values, d)
    return None


def convergence_test():
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

    run(beta_values[0], d, num_elements_in_region, total_num_elements, EpsilonR, MuR)
    # sweep_parameter(beta_values, d, num_elements_in_region, total_num_elements, EpsilonR, MuR)
    # convergence_test()

