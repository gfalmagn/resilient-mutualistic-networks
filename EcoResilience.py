import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.linalg import eigvals
from scipy.optimize import fsolve
import seaborn as sns
import networkx as nx

class GLVmodel(object):

    def __init__(self, num_species, fA=1., fM=1., fF=1., fC=1.):
        self.N = num_species
        self.fA = fA  # relative strength of antagonistic interactions
        self.fM = fM  # relative strength of mutualistic interactions
        self.fF = fF  # relative strength of facilitative interactions
        self.fC = fC  # relative strength of competitive interactions

    def generate_nested_glv_matrix(self, pA=0.2, pM=0, pF=0, pC=0, nestedness_level=0.7, nested_factor=1.5):
        """
        Generate a GLV interaction matrix with nestedness.

        Parameters:
            N: number of species
            fA: strength of antagonistic interactions
            fM: strength of mutualistic interactions
            fF: strength of facilitation
            fC: strength of competition
            pA, pM, pF, pC: density of each interaction type
            nestedness_level: proportion of core species
            nested_factor: scaling factor for core interactions

        Returns:
            interaction_matrix: interaction matrix (N x N)
        """
        # Conversion efficiency when i utilizes j in the corresponding interaction:
        # "g": antagonism; "e": mutualism; "f": facilitation; "c": competition
        const_efficiency = 0.5  # to make all efficiencies random as in the paper, use -1 here
        g = np.full((self.N, self.N), const_efficiency) if const_efficiency >= 0 else np.random.uniform(0, 1, (self.N, self.N))
        e = np.full((self.N, self.N), const_efficiency) if const_efficiency >= 0 else np.random.uniform(0, 1, (self.N, self.N))
        f = np.full((self.N, self.N), const_efficiency) if const_efficiency >= 0 else np.random.uniform(0, 1, (self.N, self.N))
        c = np.full((self.N, self.N), const_efficiency) if const_efficiency >= 0 else np.random.uniform(0, 1, (self.N, self.N))
        A = np.random.uniform(0, 1, (self.N, self.N))  # Potential interaction preferences
        np.fill_diagonal(A, 0)

        P = pA + pM + pF + pC
        assert P <= 1 + 1e-5, "Total interaction density must be less than 1, but you entered {} + {} + {} + {} = {}".format(
            pA, pM, pF, pC, P)
        if P > 1:
            pA -= 1e-5
        N_pairs = int(P * self.N * (self.N - 1) / 2)  # number of connected pairs (edges) [SH]

        # Construct list of all pairs of species (`L_max` in Mougi & Kondo, Science 2012)
        max_pairs = np.array([(i, j) for i in range(self.N) for j in range(i + 1, self.N)], dtype="i,i")

        # Pick out P links randomly
        selected_pairs = np.random.choice(max_pairs, size=N_pairs, replace=False)

        # Pick out interactions for each type
        interactions = {}
        for label, p in zip('amfc', [pA, pM, pF, pC]):
            size = int(p * len(max_pairs))
            interactions[label] = sorted(np.random.choice(selected_pairs, size=size, replace=False), key=lambda x: x[0])
            # get pairs that are in sel but not in interactions
            pairs_not_assigned = np.logical_not(np.isin(selected_pairs, interactions[label]))
            remaining_pairs = selected_pairs[pairs_not_assigned]

        interaction_matrix = np.zeros((self.N, self.N))
        # Set diagonal terms to ensure negative intraspecific interactions
        for i in range(self.N):
            interaction_matrix[i, i] = -np.random.uniform()  # Ensuring stability by negative self-regulation

        print(A)
        selected_pairs = [(1, 3), (0, 1), (1, 2), (2, 3)]
        interactions = {"a": [(2, 3), (0, 1)], "m": [(1, 3), (1, 2)], "f": [], "c": []}
        print(interactions)

        core_species = int(self.N * nestedness_level)
        for label in "amfc":
            print(f"{label}:", interactions[label])
            for edge in interactions[label]:
                i, j = edge
                # if (i < core_species or j < core_species):
                #     factor = nested_factor  # Increase factor for core interactions
                # else:
                factor = 1.0
                if label == "a":
                    # j (higher index) preys on i --> ensures directionality
                    resource_sum = 0
                    for pair in interactions[label]:
                        print(pair)
                        if pair[0] == i:
                            resource_sum += A[j, pair[0]]
                        # elif pair[1] == j:
                        #     resource_sum += A[pair[0], j]
                    print(resource_sum)
                    interaction_matrix[j, i] = factor * g[j, i] * self.fA * A[j, i] / resource_sum
                    interaction_matrix[i, j] = -interaction_matrix[j, i] / g[j, i]
        print(interaction_matrix)
                # Define higher interaction probability/strength for core species
                # Select interaction type based on proportions
                # elif interaction_type_ij == 'M':  # Mutualism (e.g., plant-pollinator)
                #     interaction_matrix[i, j] = self.fM * factor
                #     interaction_matrix[j, i] = self.fM * factor
                # elif interaction_type_ij == 'F':  # Facilitation (one-sided benefit)
                #     interaction_matrix[i, j] = self.fF * factor
                #     interaction_matrix[j, i] = 0
                # elif interaction_type_ij == 'C':  # Competition (both negative)
                #     interaction_matrix[i, j] = -self.fC * factor
                #     interaction_matrix[j, i] = -self.fC * factor

        return interaction_matrix
    
    def generate_lv_params(self, pA=0.2, pM=0, pF=0, pC=0, nestedness_level=0.7, nested_factor=1.5):
        """ Generate equilibrium point and solve for r """
    
        interaction_matrix = self.generate_nested_glv_matrix(pA, pM, pF, pC, nestedness_level, nested_factor)
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
        Sort nodes by their degree (number of interactions) in descending order.
        Parameters:
            interaction_matrix: The interaction matrix (adjacency matrix) to analyze.
        Returns:
            Sorted adjacency matrix and the order of nodes.
        """
        # Create a NetworkX graph from the adjacency matrix
        G = nx.from_numpy_array(interaction_matrix)

        # Compute degrees for each node
        degrees = dict(G.degree())

        # Sort nodes by degree (in descending order)
        sorted_nodes = sorted(degrees, key=degrees.get, reverse=True)

        # Reorder the adjacency matrix
        sorted_matrix = interaction_matrix[np.ix_(sorted_nodes, sorted_nodes)]

        return sorted_matrix, sorted_nodes

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
        sns.heatmap(sorted_matrix, cmap='coolwarm', linewidths=0.5, annot=False, square=True)
        plt.title("Adjacency Matrix Heatmap (Sorted by Degree)")
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