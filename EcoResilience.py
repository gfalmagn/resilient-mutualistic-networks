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
        """
        Generate a random symmetric adjacency matrix with a specified number of ones.

        Parameters
        ----------
        num_ones : int
            The desired number of non-zero (1) entries in the upper/lower triangle
            of the matrix (excluding diagonal).

        Returns
        -------
        numpy.ndarray
            A symmetric N x N binary adjacency matrix with exactly num_ones pairs of ones,
            where N is the size of the network (self.N). The diagonal entries are zero.

        Notes
        -----
        - The matrix is symmetric, meaning if entry (i,j) is 1, then (j,i) is also 1
        - Self-loops are not allowed (diagonal entries remain 0)
        - The total number of ones in the matrix will be 2 * num_ones due to symmetry
        """
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

        # Potential interaction preferences
        A = np.random.uniform(0, 1, (self.N, self.N))

        interaction_densities = {'antagonistic': p_a, 'mutualistic': p_m, 'facilitative': p_f, 'competitive': p_c}
        total_density = sum(interaction_densities.values())

        if not total_density <= 1:
            raise ValueError(f"Total interaction density must be <= 1, got {total_density:.3f}")

        # Construct list of all pairs of species (`L_max` in Mougi & Kondo, Science 2012)
        max_pairs = np.array([(i, j) for i in range(self.N) for j in range(i + 1, self.N)], dtype="i,i")
        L_max = len(max_pairs)

        # Pick out interactions for each type
        core_species = int(self.N * (1 - nestedness_level)) # Number of core species
        links = defaultdict(list)
        for label, p in zip('amfc', [p_a, p_m, p_f, p_c]):
            size = int(p * L_max)  # Calculate size based on the number of possible pairs
            num_selected = 0
            while num_selected < size and len(max_pairs) > 0:
                index, (i, j) = random.choice(list(enumerate(max_pairs)))
                if i < core_species or j < core_species or random.random() > nestedness_level: 
                    # If at least one of the species is a core species, add the link;
                    # otherwise, add the link with probability `nestedness_level`
                    links[label].append((i, j))
                    max_pairs = np.delete(max_pairs, index)
                    num_selected += 1

        interaction_matrix = np.zeros((self.N, self.N))
        np.fill_diagonal(interaction_matrix, -np.diag(A)) # Set diagonal terms to ensure negative intraspecific interactions

        resources = defaultdict(list)
        for label in "amfc":
            for i, j in links[label]:
                resources[i].append(j)

        factor = nested_factor  # Increase factor for core interactions
        # TODO: why is the factor constant for all interactions regardless of core or not?
        for i, j in links["a"]: # j (higher index) preys on i --> ensures directionality
            interaction_matrix[j, i] = factor * G[j, i] * self.fA * A[j, i] / np.sum(A[j, resources[j]])
            interaction_matrix[i, j] = -interaction_matrix[j, i] / G[j, i]
        for i, j in links["m"]:
            interaction_matrix[i, j] = factor * E[i, j] * self.fM * A[i, j] / np.sum(A[i, resources[i]])
            interaction_matrix[j, i] = factor * E[j, i] * self.fM * A[j, i] / np.sum(A[j, resources[j]])
        for i, j in links["f"]:
            interaction_matrix[i, j] = factor * F[i, j] * self.fF * A[i, j] / np.sum(A[i, resources[i]])
            interaction_matrix[j, i] = 0
        for i, j in links["c"]:
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
        adjacency_matrix = np.sign(abs(interaction_matrix)) # Convert to binary matrix
        degrees = adjacency_matrix.sum(axis=0) + adjacency_matrix.sum(axis=1) # Sum in and out degrees 
        # TODO: why sum both in and out degrees? Should we sum only in-degrees or out-degrees?

        # Get the sorted indices in descending order
        sorted_indices = np.argsort(degrees)[::-1]

        # Rearrange rows and columns based on sorted indices
        sorted_adjacency = adjacency_matrix[np.ix_(sorted_indices, sorted_indices)]

        return sorted_adjacency, sorted_indices

    def visualize_interaction_matrix(self, interaction_matrix):
        """
        Visualizes the adjacency (interaction) matrix as a heatmap and a network graph.
        Parameters:
            interaction_matrix: The interaction matrix (adjacency matrix) to visualize.
        """
        # Sort nodes by degree
        sorted_matrix, sorted_nodes = self.sort_nodes_by_degree(interaction_matrix)
        vmax = np.abs(sorted_matrix).max()

        # Heatmap of the interaction matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(sorted_matrix, cmap='bwr_r', linewidths=0.5, annot=False, square=True, vmin=-vmax, vmax=vmax)
        plt.title("Interaction Matrix Heatmap")
        plt.xlabel("Species")
        plt.ylabel("Species")
        plt.show()


    def visualize_adjacency_matrix(self, interaction_matrix):
        """
        Visualizes the adjacency (interaction) matrix as a heatmap and a network graph.
        Parameters:
            interaction_matrix: The interaction matrix (adjacency matrix) to visualize.
        """
        # Sort nodes by degree
        sorted_matrix, sorted_nodes = self.sort_nodes_by_degree(interaction_matrix)

        # Heatmap of the interaction matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(sorted_matrix, cmap='gray_r', linewidths=0.5, linecolor="lightgray", annot=False, square=True, cbar=False)
        plt.title("Adjacency Matrix") 
        plt.xlabel("Species")
        plt.ylabel("Species")
        plt.show()

    def extract_proportion_of_interactions(self, adj, num_total_links, interaction_type):
        """
        This function takes an adjacency matrix (eg from Kefi et al. PLOS Biology, 2016) to
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
        num_intraspecific = 0
        num_monodirectional = 0
        num_bidirectional = 0
        for i in range(adj.shape[0]):
            for j in range(i, adj.shape[1]):
                if i == j:
                    num_intraspecific += 1
                    links.append([i, j])
                else:
                    if adj[i, j] == 1 and adj[j, i] == 0:
                        num_monodirectional += 1
                        links.append([i, j])
                    elif adj[i, j] == 1 and adj[j, i] == 1:
                        num_bidirectional += 1
                        links.extend([[i, j], [j, i]])
                    elif adj[i, j] == 0 and adj[j, i] == 1:
                        num_monodirectional += 1
                        links.append([j, i])

        num_interactions = 0
        if interaction_type == 'a':  # Tropic interactions
            num_interactions = num_monodirectional

        elif interaction_type == 'f':  # Facilitating interactions
            num_interactions = num_monodirectional

        elif interaction_type == "m":
            num_interactions = num_bidirectional

        elif interaction_type == 'c':  # Competitive interactions
            num_interactions = num_bidirectional

        proportion = num_interactions / num_total_links

        return proportion