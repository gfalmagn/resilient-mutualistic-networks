import numpy as np
from numpy.linalg import eig
import networkx as nx
import matplotlib.pyplot as plt
from scipy.linalg import eigvals
from scipy.optimize import fsolve
import seaborn as sns
import networkx as nx
import random
from collections import defaultdict

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
        # const_efficiency = -1  # to make all efficiencies random as in the paper, use -1 here
        const_efficiency = 0.85  # Fix all efficiencies as in S2Table.DOCX
        G = np.full((self.N, self.N), const_efficiency) if const_efficiency > 0 else np.random.uniform(0, 1, (self.N, self.N))
        E = np.full((self.N, self.N), const_efficiency) if const_efficiency > 0 else np.random.uniform(0, 1, (self.N, self.N))
        F = np.full((self.N, self.N), const_efficiency) if const_efficiency > 0 else np.random.uniform(0, 1, (self.N, self.N))
        C = np.full((self.N, self.N), const_efficiency) if const_efficiency > 0 else np.random.uniform(0, 1, (self.N, self.N))
        # G = np.random.uniform(0, 1, (self.N, self.N))
        # E = np.random.uniform(0, 1, (self.N, self.N))
        # F = np.random.uniform(0, 1, (self.N, self.N))
        # C = np.random.uniform(0, 1, (self.N, self.N))

        # Potential interaction preferences
        A = np.random.uniform(0, 1, (self.N, self.N))
        # np.fill_diagonal(A, 0)

        P = p_a + p_m + p_f + p_c
        assert P <= 1 + 1e-5, "Total interaction density must be less than 1, but you entered {} + {} + {} + {} = {}".format(
            p_a, p_m, p_f, p_c, P)
        if P > 1:
            p_a -= 1e-5
        # N_pairs = int(P * self.N * (self.N - 1) / 2)  # number of connected pairs - REDUNDANT

        # Construct list of all pairs of species (`L_max` in Mougi & Kondo, Science 2012)
        max_pairs = np.array([(i, j) for i in range(self.N) for j in range(i + 1, self.N)], dtype="i,i")
        L_max = len(max_pairs)

        # # Pick out P links randomly
        # selected_pairs = np.random.choice(max_pairs, size=N_pairs, replace=False)
        # print(N_pairs, selected_pairs.size)

        # Pick out interactions for each type
        core_species = int(self.N * (1 - nestedness_level))
        interactions = defaultdict(list)
        for label, p in zip('amfc', [p_a, p_m, p_f, p_c]):
            size = int(p * L_max)
            num_selected = 0
            while num_selected < size and len(max_pairs) > 0:
                index, pair = random.choice(list(enumerate(max_pairs)))#selected_pairs)))
                i, j = pair[0], pair[1]
                if i < core_species or j < core_species:
                    interactions[label].append((i, j))
                    max_pairs = np.delete(max_pairs, index)
                    num_selected += 1
                else:
                    if np.random.rand() > nestedness_level:
                        interactions[label].append((i, j))
                        max_pairs = np.delete(max_pairs, index)
                        num_selected += 1

        interaction_matrix = np.zeros((self.N, self.N))
        # Set diagonal terms to ensure negative intraspecific interactions
        for i in range(self.N):
            interaction_matrix[i, i] = -A[i, i]  #-np.random.uniform()  # Ensuring stability by negative self-regulation

        resources = {}  #defaultdict(list)
        for i in range(self.N):
            resources[i] = [i]
        for label in "amfc":
            for i, j in interactions[label]:
                resources[i].append(j)

        factor = nested_factor  # Increase factor for core interactions
        for label in "amfc":
            if label == "a":
                # j (higher index) preys on i --> ensures directionality
                for i, j in interactions[label]:
                    interaction_matrix[j, i] = factor * G[j, i] * self.fA * A[j, i] / np.sum(A[j, resources[j]])
                    interaction_matrix[i, j] = -interaction_matrix[j, i] / G[j, i]
            elif label == "m":
                for i, j in interactions[label]:
                    interaction_matrix[i, j] = factor * E[i, j] * self.fM * A[i, j] / np.sum(A[i, resources[i]])
                    interaction_matrix[j, i] = factor * E[j, i] * self.fM * A[j, i] / np.sum(A[j, resources[j]])
            elif label == "f":
                for i, j in interactions[label]:
                    interaction_matrix[i, j] = factor * F[i, j] * self.fF * A[i, j] / np.sum(A[i, resources[i]])
                    interaction_matrix[j, i] = 0
            elif label == "c":
                for i, j in interactions[label]:
                    interaction_matrix[i, j] = -factor * C[i, j] * self.fC * A[i, j] / np.sum(A[i, resources[i]])
                    interaction_matrix[j, i] = -factor * C[j, i] * self.fC * A[j, i] / np.sum(A[j, resources[j]])

        return interaction_matrix

    
    def generate_glv_params(self, p_a=0.2, p_m=0, p_f=0, p_c=0, nestedness_level=0.7, nested_factor=1.5):
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

    def check_stability(self, interaction_matrix, X_eq):
        # Calculate M, the community matrix, i.e. the Jacobian at X_eq
        # For the GLV equations this is very simple
        # M_ij = X_eq_i & a_ij
        M = X_eq * interaction_matrix
        eigenvalues, _ = eig(M)

        return np.all(np.real(eigenvalues) < 0)

    # def compute_jacobian(self, interaction_matrix, X_eq, r):
    #     """
    #     Compute the Jacobian matrix at equilibrium X_eq for assessing stability.
    #     """
    #     J = np.zeros((self.N, self.N))
    #     for i in range(self.N):
    #         for j in range(self.N):
    #             if i == j:
    #                 J[i, j] = r[i] - np.sum(interaction_matrix[i, :] * X_eq)
    #             else:
    #                 J[i, j] = interaction_matrix[i, j] * X_eq[j]
    #
    #     return J
    #
    # def check_stability(self, jacobian_matrix, plot=False):
    #     eigenvalues = eigvals(jacobian_matrix)
    #     print(eigenvalues)
    #     real_parts = eigenvalues.real
    #
    #     if plot:
    #         # Plot eigenvalues for visualization
    #         plt.scatter(real_parts, eigenvalues.imag, color='blue')
    #         plt.axvline(0, color='red', linestyle='--')
    #         plt.xlabel('Real Part')
    #         plt.ylabel('Imaginary Part')
    #         plt.title('Eigenvalue Spectrum of Jacobian Matrix')
    #         plt.show()
    #
    #     # Check if all eigenvalues have negative real parts
    #     stable = np.all(real_parts < 0)
    #     return stable, eigenvalues

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
        Sort the adjacency matrix A based on the degrees of the nodes.

        Parameters:
            A (np.ndarray): Adjacency matrix (N x N).

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
                # if i == j:
                #     adjacency_matrix[i, j] = 0
        # Compute degrees (sum of rows + columns for undirected graphs)
        degrees = np.sum(adjacency_matrix, axis=0) + np.sum(adjacency_matrix, axis=1)

        # Get the sorted indices in descending order
        sorted_indices = np.argsort(degrees)[::-1]

        # Rearrange rows and columns based on sorted indices
        sorted_adjacency = adjacency_matrix[np.ix_(sorted_indices, sorted_indices)]
        # sorted_interactions = interaction_matrix[np.ix_(sorted_indices, sorted_indices)]

        return sorted_adjacency, sorted_indices

    def visualize_interaction_matrix(self, interaction_matrix):
        """
        Visualizes the adjacency (interaction) matrix as a heatmap and a network graph.
        Parameters:
            interaction_matrix: The interaction matrix (adjacency matrix) to visualize.
        """
        # Sort nodes by degree
        sorted_matrix, sorted_nodes = self.sort_nodes_by_degree(interaction_matrix)
        # print(sorted_matrix)
        vmax = np.abs(sorted_matrix).max()

        # Heatmap of the interaction matrix
        plt.figure(figsize=(10, 8))
        # sns.heatmap(np.abs(sorted_matrix), cmap='gray_r', linewidths=0.5, annot=False, square=True)
        sns.heatmap(sorted_matrix, cmap='bwr_r', linewidths=0.5, annot=False, square=True, vmin=-vmax, vmax=vmax)
        plt.title("Interaction Matrix Heatmap")# (Sorted by Degree)")
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
        sns.heatmap(sorted_matrix, cmap='gray_r', linewidths=0.5, linecolor="lightgray", annot=False, square=True, cbar=False)
        # sns.heatmap(sorted_matrix, cmap='coolwarm', linewidths=0.5, annot=False, square=True)
        plt.title("Adjacency Matrix")  # Heatmap")# (Sorted by Degree)")
        plt.xlabel("Species")
        plt.ylabel("Species")
        plt.show()

    def extract_proportion_of_interactions(self, adj, num_total_links, interaction_type):
        """
        This function takes an adjacency matrix (from Kefi et al. PLOS Biology, 2016) to
        randomize the entries a_ij, which represent interaction strengths between species i and j.

        :param
            adj (array): adjacency matrix with entries 0 and 1 only
            type (str): Interaction types ('a': antagonism = trophic interaction;
                        'f': facilitation = non-trophic positive interaction;
                        'c': competition = non-trophic negative interaction)
            strength (float): interaction strength
        :return:
            a_rand: randomized interaction matrix
        """
        links = []
        num_links = 0
        num_intraspecific = 0
        num_monodirectional = 0
        num_bidirectional = 0
        for i in range(adj.shape[0]):
            for j in range(i, adj.shape[1]):
                if i == j:
                    num_intraspecific += 1
                    links.append([i, j])
                    # num_links += 1
                else:
                    if adj[i, j] == 1 and adj[j, i] == 0:
                        num_monodirectional += 1
                        links.append([i, j])
                        num_links += 1
                    elif adj[i, j] == 1 and adj[j, i] == 1:
                        num_bidirectional += 1
                        links.append([i, j])
                        links.append([j, i])
                        num_links += 2
                    elif adj[i, j] == 0 and adj[j, i] == 1:
                        num_monodirectional += 1
                        links.append([j, i])
                        num_links += 1
        # print(f"{interaction_type}: {num_links}")
        # print(f"intraspecific: {num_intraspecific}, mono-dir: {num_monodirectional}, bi-dir: {num_bidirectional}")

        num_interactions = 0
        if interaction_type == 'a':  # Tropic interactions
            num_interactions = num_monodirectional
            # print(f"{interaction_type}: {num_interactions}")

        elif interaction_type == 'f':  # Facilitating interactions
            num_interactions = num_monodirectional
            # print(f"{interaction_type}: {num_interactions}")

        elif interaction_type == "m":
            num_interactions = num_bidirectional
            # print(f"{interaction_type}: {num_interactions}")

        elif interaction_type == 'c':  # Competitive interactions
            num_interactions = num_bidirectional
            # print(f"{interaction_type}: {num_interactions}")

        proportion = num_interactions / num_total_links
        # print(f"{num_interactions} links lead to p_{interaction_type} = {proportion:.4f}")

        return proportion