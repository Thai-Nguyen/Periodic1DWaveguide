'''
assembly

'''
import numpy as np

nodes_per_element = 2


def __which_region(e, num_elements):
    '''
    Determines whether the element e is in region 0 or 1.
    '''

    if e >= 0 and e <= num_elements[0]:
        k = 0  # element in region 1
    elif e > num_elements[0] and (e <= num_elements[0] + num_elements[1]):
        k = 1  # element in region 2
    else:
        print('Error: element not in any region')
        return None
    return k


def __create_connectivity_array(nodes_per_element, total_num_elements):
    '''
    Method for creating connectivity array; an array used to assign a global
      node number for local node of a certain element
    :param nodes_per_element:
    :param total_num_elements:
    :return:
    '''

    n = np.zeros((nodes_per_element, total_num_elements), dtype='int16')

    # Create connectivity array
    node_num = 0
    for e in range(0, total_num_elements):  # row
        for i in range(0, nodes_per_element):  # col
            n[(i, e)] = node_num
            node_num += 1
        node_num -= 1
    # Apply periodic boundary condition
    n[nodes_per_element-1, total_num_elements-1] = 0
    return n


def assembly(num_elements, total_num_elements, le, beta, EpsilonR, MuR):
    '''
    Assembles the A and B matrices to be used to solve the eigenvalue problem
      for the  1D periodic structure
    :param num_elements:
    :param total_num_elements:
    :param le:
    :param beta:
    :param EpsilonR:
    :param MuR:
    :return:
    '''
    global nodes_per_element

    # Pre-allocate memory
    elG = np.zeros((nodes_per_element, nodes_per_element), dtype='complex128')
    elT = np.zeros((nodes_per_element, nodes_per_element), dtype='complex128')
    elD = np.zeros((nodes_per_element, nodes_per_element), dtype='complex128')
    elB = np.zeros((nodes_per_element, nodes_per_element), dtype='complex128')

    A = np.zeros((total_num_elements, total_num_elements), dtype='complex128')
    B = np.zeros((total_num_elements, total_num_elements), dtype='complex128')

    # Make connectivity array
    n = __create_connectivity_array(nodes_per_element, total_num_elements)

    # Fill matrices A and B
    for e in range(0, total_num_elements):
        # Determine which region the element e is in
        k = __which_region(e, num_elements)

        # Calculate the components of the element matrix elG
        elG[(0, 0)] = 1 / (le[k] * MuR)
        elG[(0, 1)] = -1 / (le[k] * MuR)
        elG[(1, 0)] = -1 / (le[k] * MuR)
        elG[(1, 1)] = 1 / (le[k] * MuR)

        # Calculate the components of the element matrix elT
        elT[(0, 0)] = (beta ** 2 * le[k]) / (3 * MuR)
        elT[(0, 1)] = (beta ** 2 * le[k]) / (6 * MuR)
        elT[(1, 0)] = (beta ** 2 * le[k]) / (6 * MuR)
        elT[(1, 1)] = (beta ** 2 * le[k]) / (3 * MuR)

        # Calculate the components of the element matrix elD
        elD[(0, 0)] = 0
        elD[(0, 1)] = np.complex(0, beta) / MuR
        elD[(1, 0)] = np.complex(0, -beta) / MuR
        elD[(1, 1)] = 0

        # Calculate the components of the element matrix elB
        elB[(0, 0)] = (le[k] * EpsilonR[k]) / 3
        elB[(0, 1)] = (le[k] * EpsilonR[k]) / 6
        elB[(1, 0)] = (le[k] * EpsilonR[k]) / 6
        elB[(1, 1)] = (le[k] * EpsilonR[k]) / 3

        # Add elements to A and B matrices
        for i in range(0, nodes_per_element):
            for j in range(0, nodes_per_element):
                A[n[i, e], n[j, e]] += elG[i, j] + elT[i, j] + elD[i, j]
                B[n[i, e], n[j, e]] += elB[i, j]
    return A, B
