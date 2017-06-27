import numpy as np
import matplotlib.pyplot as plt
from assembly import assembly
from solver import solver

# Global parameters
max_iterations = 100

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
        plt.plot(beta_values * d, np.real(k0[i, :]), '.')
    plt.grid()
    plt.xlim(0, np.pi)
    plt.ylim(0, 25)
    plt.xlabel('$\\beta d$')
    plt.ylabel('$k_0$')
    plt.show()
    return None


def plot_electric_field(eig_value, eig_vector, beta, d):
    # Plot the first three modes
    for i in range(5):
        print('k0:', np.sqrt(eig_value[i]))
        plt.plot(np.real(eig_vector[:, i]))
    plt.xlabel('n')
    plt.ylabel('E')
    plt.grid()
    plt.show()
    return None


def plot_convergence(min_k0):
    plt.plot(min_k0)
    plt.show()
    return None


def run(beta, d, num_elements_in_region, total_num_elements, EpsilonR, MuR):
    le = preprocess()
    eig_value, eig_vector = fem(le, beta, num_elements_in_region, total_num_elements, EpsilonR, MuR)
    plot_electric_field(eig_value, eig_vector, beta, d)
    return None


def sweep_parameter(beta_values, d, num_elements_in_region, total_num_elements, EpsilonR, MuR):
    le = preprocess()
    # Allocate space for eigenvalues
    all_eig = np.zeros((total_num_elements, beta_values.size), dtype='complex128')

    # Sweep beta value and find corresponding k0
    print(np.size(beta_values), 'values to sweep')
    print('Start:', beta_values[0], 'End:', beta_values[beta_values.size-1])
    for i in range(beta_values.size):
        print('iter:', i+1, '/ beta:', beta_values[i])
        all_eig[:, i], _ = fem(le, beta_values[i], num_elements_in_region, total_num_elements, EpsilonR, MuR)
    plot_dispersion(all_eig, beta_values, d)
    return None


def convergence_test(beta, EpsilonR, MuR):
    # Allocate memory for smallest k0
    min_k0 = np.zeros((max_iterations, 1))
    le = preprocess()

    # Run convergence test
    for i in range(1, max_iterations):
        num_elements_in_region = np.array((10*i, 10*i))
        total_num_elements = np.sum(num_elements_in_region)
        eig_value, eig_vector = fem(le, beta, num_elements_in_region, total_num_elements, EpsilonR, MuR)

        min_k0[i] = np.sqrt(eig_value[0])
    plot_convergence(min_k0)
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

    # run(beta_values[10], d, num_elements_in_region, total_num_elements, EpsilonR, MuR)
    # sweep_parameter(beta_values, d, num_elements_in_region, total_num_elements, EpsilonR, MuR)
    convergence_test(beta_values[10], EpsilonR, MuR)
