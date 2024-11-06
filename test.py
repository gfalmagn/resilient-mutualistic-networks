import numpy as np
from scipy.optimize import fsolve
from scipy.linalg import eigvals
import networkx as nx
import matplotlib.pyplot as plt

# Step 1: Generate GLV interaction matrix with specific interaction types
def generate_glv_matrix_cascade(N, fA, fM, fF, fC, pA, pM, pF, pC):
    interaction_matrix = np.zeros((N, N))

    for i in range(N):
        for j in range(i + 1, N):
            interaction_type = np.random.choice(
                ['A', 'M', 'F', 'C'], p=[pA, pM, pF, pC]
            )
            if interaction_type == 'A':  # Antagonism (e.g., predator-prey)
                interaction_matrix[i, j] = -fA
                interaction_matrix[j, i] = fA
            elif interaction_type == 'M':  # Mutualism (e.g., plant-pollinator)
                interaction_matrix[i, j] = fM
                interaction_matrix[j, i] = fM
            elif interaction_type == 'F':  # Facilitation (one-sided benefit)
                interaction_matrix[i, j] = fF
                interaction_matrix[j, i] = 0
            elif interaction_type == 'C':  # Competition (both negative)
                interaction_matrix[i, j] = -fC
                interaction_matrix[j, i] = -fC
    return interaction_matrix

# Step 2: Generate equilibrium point and solve for r
def generate_lv_params(N, pA=0.1, pM=0, pF=0, pC=0):
    fA = 1  # Effect strength for antagonism
    fM = 1  # Effect strength for mutualism
    fF = 1  # Effect strength for facilitation
    fC = 1  # Effect strength for competition

    interaction_matrix = generate_glv_matrix_cascade(N, fA, fM, fF, fC, pA, pM, pF, pC)
    X_eq = np.random.uniform(0, 1, N)  # Random equilibrium abundances

    # Define function to find r that satisfies the equilibrium condition
    def glv_r(r):
        return X_eq * (r + interaction_matrix @ X_eq)

    # Initial guess for growth rates, r
    r0 = np.ones(N) / 2
    r = fsolve(glv_r, r0)

    return r, interaction_matrix, X_eq

# Step 3: Calculate the Jacobian matrix at equilibrium
def compute_jacobian(interaction_matrix, X_eq, r):
    N = len(r)
    jacobian_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i == j:
                jacobian_matrix[i, j] = r[i] - X_eq[i] * interaction_matrix[i, j]
            else:
                jacobian_matrix[i, j] = -X_eq[i] * interaction_matrix[i, j]
    return jacobian_matrix

# Step 4: Check stability using eigenvalues of the Jacobian matrix
def check_stability(jacobian_matrix, plot=False):
    eigenvalues = eigvals(jacobian_matrix)
    real_parts = eigenvalues.real

    if plot:
        # Plot eigenvalues for visualization
        plt.scatter(real_parts, eigenvalues.imag, color='blue')
        plt.axvline(0, color='red', linestyle='--')
        plt.xlabel('Real Part')
        plt.ylabel('Imaginary Part')
        plt.title('Eigenvalue Spectrum of Jacobian Matrix')
        plt.show()

    # Check if all eigenvalues have negative real parts
    stable = np.all(real_parts < 0)
    return stable, eigenvalues

# Parameters
N = 100  # Number of species
pA, pM, pF, pC = 0.3, 0.4, 0., 0.  # Interaction type proportions

stability = []
for _ in range(100):
    # Generate parameters and assess stability
    r, interaction_matrix, X_eq = generate_lv_params(N, pA, pM, pF, pC)

    jacobian_matrix = compute_jacobian(interaction_matrix, X_eq, r)
    stable, eigenvalues = check_stability(jacobian_matrix)

    # print("Equilibrium abundances:", X_eq)
    # print("Is the network stable?", stable)
    stability.append(stable)
print(sum(stability))