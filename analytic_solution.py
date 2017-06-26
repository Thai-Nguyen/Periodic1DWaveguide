'''
Analytic solution to periodic 1D structure

Calculate dispersion relation and plots k0 vs beta

Thai Nguyen
June 21, 2017
'''
import matplotlib.pyplot as plt
import numpy as np

def betad(k0, n1, n2, d1, d2):
    # RHS of dispersion relation
    rhs = np.cos(k0 * n1 * d1) * np.cos(k0 * n2 * d2) \
             - (n1**2 + n2**2) / (2*n1*n2) * np.sin(k0 * n1 * d1) * np.sin(k0 * n2 * d2)
    return np.arccos( rhs )

# Structure parameters
n1 = 3 #index of refraction in region 1
n2 = 1

d1 = 100e-3 #width of region 1
d2 = d1
d = d1 + d2 #width of entire cell

# Parameter to sweep
k0_values = np.linspace(0, 25, 1e5)

# Calculate beta
beta_values = betad(k0_values,n1,n2,d1,d2)

# Plot
plt.plot(beta_values, k0_values)
plt.grid()
plt.xlim(0,np.pi)
plt.xlabel(r'$\beta d$')
plt.ylabel(r'$k_0$')
plt.show()