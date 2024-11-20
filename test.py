import numpy as np
from scipy.optimize import fsolve
from scipy.linalg import eigvals
import networkx as nx
import matplotlib.pyplot as plt

class GLVmodel(object):

    def __init__(self, num_species, fA=1., fM=1., fF=1., fC=1.):
        self.N = num_species
        self.fA = fA  # relative strength of antagonistic interactions
        self.fM = fM  # relative strength of mutualistic interactions
        self.fF = fF  # relative strength of facilitative interactions
        self.fC = fC  # relative strength of competitive interactions

    # # Generate GLV interaction matrix with specific interaction types
    # def generate_glv_matrix_cascade(self, pA, pM, pF, pC):
    #     interaction_matrix = np.zeros((N, N))
    #
    #     for i in range(N):
    #         for j in range(i + 1, N):
    #             interaction_type = np.random.choice(
    #                 ['A', 'M', 'F', 'C'], p=[pA, pM, pF, pC]
    #             )
    #             if interaction_type == 'A':  # Antagonism (e.g., predator-prey)
    #                 interaction_matrix[i, j] = -fA
    #                 interaction_matrix[j, i] = fA
    #             elif interaction_type == 'M':  # Mutualism (e.g., plant-pollinator)
    #                 interaction_matrix[i, j] = fM
    #                 interaction_matrix[j, i] = fM
    #             elif interaction_type == 'F':  # Facilitation (one-sided benefit)
    #                 interaction_matrix[i, j] = fF
    #                 interaction_matrix[j, i] = 0
    #             elif interaction_type == 'C':  # Competition (both negative)
    #                 interaction_matrix[i, j] = -fC
    #                 interaction_matrix[j, i] = -fC
    #     return interaction_matrix

    # Generate GLV (generalized Lotka-Volterra) interaction matrix with nestedness
    def generate_nested_glv_matrix(self, pA, pM, pF, pC, nestedness_level=0.7):
        # Core and peripheral indices based on nestedness level
        core_species = int(N * nestedness_level)
        interaction_matrix = np.zeros((N, N))

        e = np.random.uniform(0, 1, (self.N, self.N))
        g = np.random.uniform(0, 1, (self.N, self.N))
        f = np.random.uniform(0, 1, (self.N, self.N))
        c = np.random.uniform(0, 1, (self.N, self.N))

        for i in range(N):
            for j in range(i + 1, N):
                # Define higher interaction probability/strength for core species
                if (i < core_species) or (j < core_species):
                    factor = 1.5  # Increase factor for core interactions
                else:
                    factor = 1.0

                # Select interaction type based on proportions
                interaction_type = np.random.choice(['A', 'M', 'F', 'C'], p=[pA, pM, pF, pC])

                # Assign interaction strength based on type and nestedness factor
                if interaction_type == 'A':  # Antagonism (e.g., predator-prey)
                    interaction_matrix[i, j] = self.fA * factor
                    interaction_matrix[j, i] = -self.fA * factor
                elif interaction_type == 'M':  # Mutualism (e.g., plant-pollinator)
                    interaction_matrix[i, j] = self.fM * factor
                    interaction_matrix[j, i] = self.fM * factor
                elif interaction_type == 'F':  # Facilitation (one-sided benefit)
                    interaction_matrix[i, j] = self.fF * factor
                    interaction_matrix[j, i] = 0
                elif interaction_type == 'C':  # Competition (both negative)
                    interaction_matrix[i, j] = -self.fC * factor
                    interaction_matrix[j, i] = -self.fC * factor
        return interaction_matrix


    # Generate equilibrium point and solve for r
    def generate_glv_parameters(self, pA=0.1, pM=0, pF=0, pC=0, nestedness_level=0):
        interaction_matrix = self.generate_nested_glv_matrix(pA, pM, pF, pC, nestedness_level)
        X_eq = np.random.uniform(0, 1, self.N)  # Random equilibrium abundances

        # Define function to find r that satisfies the equilibrium condition
        def glv_r(r):
            return X_eq * (r + interaction_matrix @ X_eq)

        # Initial guess for growth rates, r
        r0 = np.ones(self.N) / 2
        r = fsolve(glv_r, r0)

        return r, interaction_matrix, X_eq

    # Calculate the Jacobian matrix at equilibrium
    def compute_jacobian(self, interaction_matrix, X_eq, r):
        jacobian_matrix = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(self.N):
                if i == j:
                    jacobian_matrix[i, j] = r[i] - X_eq[i] * interaction_matrix[i, j]
                else:
                    jacobian_matrix[i, j] = -X_eq[i] * interaction_matrix[i, j]
        return jacobian_matrix

    # Check stability using eigenvalues of the Jacobian matrix
    def check_stability(self, jacobian_matrix, plot=False):
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

# # Parameters
# N = 100  # Number of species
# pA, pM, pF, pC = 0.3, 0.4, 0.1, 0.2  # Interaction type proportions
# nestedness_level = 0
#
# stability = []
# for _ in range(100):
#     # Generate parameters and assess stability
#     r, interaction_matrix, X_eq = generate_glv_parameters(N, pA, pM, pF, pC, nestedness_level)
#
#     jacobian_matrix = compute_jacobian(interaction_matrix, X_eq, r)
#     stable, eigenvalues = check_stability(jacobian_matrix)
#
#     # print("Equilibrium abundances:", X_eq)
#     # print("Is the network stable?", stable)
#     stability.append(stable)
# print(sum(stability))

N = 6
pA, pM, pF, pC = 0.3, 0.2, 0.1, 0.
P = pA + pM + pF + pC
n_a = int(P * N * (N - 1) / 2)
# pairs = np.array([(i, j) for i in range(N) for j in range(i + 1, N)], dtype="i,i")
# # print(pairs)
# sel = np.random.choice(pairs, size=n_a, replace=False)
# print(f"Randomly selected {n_a} links:", sel)
#
# # Pick out interactions for each type
# interactions = {}
# for label, p in zip('amfc', [pA, pM, pF, pC]):
#     size = int(p * n_a)
#     interactions[label] = list(np.random.choice(sel, size=size, replace=False))
#     # # get pairs that are in sel but not in interactions
#     # remaining_pairs = np.logical_not(np.isin(sel, interactions[label]))
#     # sel = sel[remaining_pairs]
# print(interactions)
# # print(sel)
sel =
print(f"Randomly selected {n_a} links:",
interactions = {'a': [(0, 4), (1, 3)], 'm': [(2, 3)], 'f': [], 'c': []}