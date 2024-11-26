import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.linalg import eigvals
from scipy.optimize import fsolve
import seaborn as sns
import networkx as nx
import random

class GLVmodel(object):

    def __init__(self, num_species, fA=1., fM=1., fF=1., fC=1.):
        self.N = num_species
        self.fA = fA  # relative strength of antagonistic interactions
        self.fM = fM  # relative strength of mutualistic interactions
        self.fF = fF  # relative strength of facilitative interactions
        self.fC = fC  # relative strength of competitive interactions

    def generate_random_adjacency_matrix(self, num_ones):
        adjacency_matrix = np.zeros((self.N, self.N), dtype=int)
        links_added = 0
        while links_added < num_ones:
            i, j = random.randint(0, self.N - 1), random.randint(0, self.N - 1)
            if i != j and adjacency_matrix[i, j] == 0:
                adjacency_matrix[i, j] = 1
                adjacency_matrix[j, i] = 1
                links_added += 1
        return adjacency_matrix

    def generate_nested_glv_matrix(self, p_a=0.2, p_m=0, p_f=0, p_c=0, nestedness_level=0.7, nested_factor=1.5):
        """
        Generate a GLV interaction matrix with nestedness.

        Parameters:
            N: number of species
            fA: strength of antagonistic interactions
            fM: strength of mutualistic interactions
            fF: strength of facilitation
            fC: strength of competition
            p_a, p_m, p_f, p_c: density of each interaction type
            nestedness_level: proportion of core species
            nested_factor: scaling factor for core interactions

        Returns:
            interaction_matrix: interaction matrix (N x N)
        """
        # Conversion efficiency when i utilizes j in the corresponding interaction:
        # "g": antagonism; "e": mutualism; "f": facilitation; "c": competition
        const_efficiency = 0.5  # to make all efficiencies random as in the paper, use -1 here
        G = np.full((self.N, self.N), const_efficiency) if const_efficiency >= 0 else np.random.uniform(0, 1, (self.N, self.N))
        E = np.full((self.N, self.N), const_efficiency) if const_efficiency >= 0 else np.random.uniform(0, 1, (self.N, self.N))
        F = np.full((self.N, self.N), const_efficiency) if const_efficiency >= 0 else np.random.uniform(0, 1, (self.N, self.N))
        C = np.full((self.N, self.N), const_efficiency) if const_efficiency >= 0 else np.random.uniform(0, 1, (self.N, self.N))
        A = np.random.uniform(0, 1, (self.N, self.N))  # Potential interaction preferences
        np.fill_diagonal(A, 0)

        P = p_a + p_m + p_f + p_c
        assert P <= 1 + 1e-5, "Total interaction density must be less than 1, but you entered {} + {} + {} + {} = {}".format(
            p_a, p_m, p_f, p_c, P)
        if P > 1:
            p_a -= 1e-5
        N_pairs = int(P * self.N * (self.N - 1) / 2)  # number of connected pairs (edges) [SH]

        # Construct list of all pairs of species (`L_max` in Mougi & Kondo, Science 2012)
        max_pairs = np.array([(i, j) for i in range(self.N) for j in range(i + 1, self.N)], dtype="i,i")

        # Pick out P links randomly
        selected_pairs = np.random.choice(max_pairs, size=N_pairs, replace=False)

        # Pick out interactions for each type
        interactions = {}
        for label, p in zip('amfc', [p_a, p_m, p_f, p_c]):
            size = int(p * len(max_pairs))
            interactions[label] = sorted(np.random.choice(selected_pairs, size=size, replace=False), key=lambda x: x[0])
            # get pairs that are in selected_pairs but not in interactions
            if interactions[label]:
                pairs_not_assigned = np.logical_not(np.isin(selected_pairs, interactions[label]))
                selected_pairs = selected_pairs[pairs_not_assigned]

        interaction_matrix = np.zeros((self.N, self.N))
        # Set diagonal terms to ensure negative intraspecific interactions
        for i in range(self.N):
            interaction_matrix[i, i] = -np.random.uniform()  # Ensuring stability by negative self-regulation

        # print(A)
        # selected_pairs = [(1, 3), (0, 1), (1, 2), (2, 3)]
        # interactions = {"a": [(2, 3), (0, 1)], "m": [(1, 3), (1, 2)], "f": [], "c": []}
        # print(interactions)

        core_species = int(self.N * nestedness_level)
        for label in "amfc":
            # print(f"{label}:", interactions[label])
            if label == "a":
                # j (higher index) preys on i --> ensures directionality
                resource_sum = 0
                for i, j in interactions[label]:
                    resource_sum += A[j, i]
                for i, j in interactions[label]:
                    if i < core_species or j < core_species:
                        factor = nested_factor  # Increase factor for core interactions
                    else:
                        if np.random.rand() > 0.5:
                            factor = 1.0
                        else:
                            factor = 0
                    interaction_matrix[j, i] = factor * G[j, i] * self.fA * A[j, i] / resource_sum
                    interaction_matrix[i, j] = -interaction_matrix[j, i] / G[j, i]
            elif label == "m":
                resource_sum_i = 0
                resource_sum_j = 0
                for i, j in interactions[label]:
                    resource_sum_i += A[i, j]
                    resource_sum_j += A[j, i]
                for i, j in interactions[label]:
                    if i < core_species or j < core_species:
                        factor = nested_factor  # Increase factor for core interactions
                    else:
                        if np.random.rand() > 0.5:
                            factor = 1.0
                        else:
                            factor = 0
                    interaction_matrix[i, j] = factor * E[i, j] * self.fM * A[i, j] / resource_sum_i
                    interaction_matrix[j, i] = factor * E[j, i] * self.fM * A[j, i] / resource_sum_j
            elif label == "f":
                resource_sum_i = 0
                resource_sum_j = 0
                for i, j in interactions[label]:
                    resource_sum_i += A[i, j]
                    resource_sum_j += A[j, i]
                for i, j in interactions[label]:
                    if i < core_species or j < core_species:
                        factor = nested_factor  # Increase factor for core interactions
                    else:
                        if np.random.rand() > 0.5:
                            factor = 1.0
                        else:
                            factor = 0
                    interaction_matrix[i, j] = factor * F[i, j] * self.fF * A[i, j] / resource_sum_i
                    interaction_matrix[j, i] = 0
            elif label == "c":
                resource_sum_i = 0
                resource_sum_j = 0
                for i, j in interactions[label]:
                    resource_sum_i += A[i, j]
                    resource_sum_j += A[j, i]
                for i, j in interactions[label]:
                    if i < core_species or j < core_species:
                        factor = nested_factor  # Increase factor for core interactions
                    else:
                        if np.random.rand() > 0.5:
                            factor = 1.0
                        else:
                            factor = 0
                    interaction_matrix[i, j] = -factor * C[i, j] * self.fC * A[i, j] / resource_sum_i
                    interaction_matrix[j, i] = -factor * C[j, i] * self.fC * A[j, i] / resource_sum_j
        # print(interaction_matrix)
        return interaction_matrix

    
    def generate_lv_params(self, p_a=0.2, p_m=0, p_f=0, p_c=0, nestedness_level=0.7, nested_factor=1.5):
        """ Generate equilibrium point and solve for r """
    
        interaction_matrix = self.generate_nested_glv_matrix(p_a, p_m, p_f, p_c, nestedness_level, nested_factor)
        X_eq = np.random.uniform(0, 1, self.N)  # Random equilibrium abundances
    
        # Define function to find r that satisfies the equilibrium condition
        def glv_r(r):
            return X_eq * (r + interaction_matrix @ X_eq)
    
        # Initial guess for growth rates, r
        r0 = np.ones(self.N) / 2
        r = fsolve(glv_r, r0)

        return r, interaction_matrix, X_eq

    def compute_jacobian(self, interaction_matrix, X_eq, r):
        """
        Compute the Jacobian matrix at equilibrium X_eq for assessing stability.
        """
        J = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(self.N):
                if i == j:
                    J[i, j] = r[i] - np.sum(interaction_matrix[i, :] * X_eq)
                else:
                    J[i, j] = interaction_matrix[i, j] * X_eq[j]

        return J

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

    def visualize_network(self, interaction_matrix):
        G = nx.from_numpy_array(interaction_matrix)
        num_nodes = len(interaction_matrix)
        colors = ["skyblue" if i < num_nodes * 0.7 else "lightgreen" for i in range(num_nodes)]
        plt.figure(figsize=(8, 8))
        pos = nx.spring_layout(G, seed=42)
        nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=300)
        nx.draw_networkx_edges(G, pos, alpha=0.5)
        nx.draw_networkx_labels(G, pos, font_size=10)
        plt.title("Ecological Network with Nested Interactions")
        plt.show()

    def sort_nodes_by_degree(self, interaction_matrix):
        """
        Sort the adjacency matrix A from interaction_matrix based on the degrees of the nodes.

        Parameters:
            interaction_matrix (np.ndarray): Interaction matrix (N x N).

        Returns:
            np.ndarray: Sorted adjacency matrix.
        """
        adjacency_matrix = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(self.N):
                if interaction_matrix[i, j] == 0:
                    adjacency_matrix[i, j] = 0
                else:
                    adjacency_matrix[i, j] = 1
                if i == j:
                    adjacency_matrix[i, j] = 0
        # # Compute degrees for each node
        # G = nx.from_numpy_array(adjacency_matrix)
        # node_degree = dict(G.degree())
        # print(node_degree)

        # Compute degrees (sum of rows + columns for undirected graphs)
        degrees = np.sum(adjacency_matrix, axis=0) + np.sum(adjacency_matrix, axis=1)

        # Get the sorted indices in descending order
        sorted_indices = np.argsort(degrees)[::-1]

        # Rearrange rows and columns based on sorted indices
        # sorted_adjacency = adjacency_matrix[np.ix_(sorted_indices, sorted_indices)]
        sorted_matrix = interaction_matrix[np.ix_(sorted_indices, sorted_indices)]

        return sorted_matrix, sorted_indices

    def visualize_adjacency_matrix(self, interaction_matrix):
        """
        Visualizes the adjacency (interaction) matrix as a heatmap and a network graph.
        Parameters:
            interaction_matrix: The interaction matrix (adjacency matrix) to visualize.
        """
        # Sort nodes by degree
        sorted_matrix, sorted_nodes = self.sort_nodes_by_degree(interaction_matrix)
        # print(sorted_matrix)

        # Heatmap of the interaction matrix
        plt.figure(figsize=(10, 8))
        # sns.heatmap(np.abs(sorted_matrix), cmap='gray_r', linewidths=0.5, annot=False, square=True)
        sns.heatmap(sorted_matrix, cmap='coolwarm', linewidths=0.5, annot=False, square=True)
        plt.title("Adjacency Matrix")# Heatmap")# (Sorted by Degree)")
        plt.xlabel("Species")
        plt.ylabel("Species")
        plt.show()

        # # Create a NetworkX graph from the interaction matrix
        # G = nx.from_numpy_array(interaction_matrix)
        #
        # # Plot the graph
        # plt.figure(figsize=(10, 10))
        # pos = nx.spring_layout(G)  # Positions for all nodes
        # nx.draw(G, pos, with_labels=True, node_size=500, node_color='lightblue', font_size=10, font_weight='bold',
        #         edge_color='gray')
        # plt.title("Network Graph of Species Interactions")
        # plt.show()