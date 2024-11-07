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
            a: the interaction matrix (N x N)
        """
        const_efficiency = 0.5
        e = np.full((self.N, self.N), const_efficiency) if const_efficiency >= 0 else np.random.uniform(0, 1, (self.N, self.N))
        g = np.full((self.N, self.N), const_efficiency) if const_efficiency >= 0 else np.random.uniform(0, 1, (self.N, self.N))
        f = np.full((self.N, self.N), const_efficiency) if const_efficiency >= 0 else np.random.uniform(0, 1, (self.N, self.N))
        c = np.full((self.N, self.N), const_efficiency) if const_efficiency >= 0 else np.random.uniform(0, 1, (self.N, self.N))
        A = np.random.uniform(0, 1, (self.N, self.N))  # Potential interaction preferences
        np.fill_diagonal(A, 0)

        core_species = int(self.N * nestedness_level)
        P = pA + pF + pC + pM
        # assert P <= 1 + 1e-5, f"Total interaction density must be <= 1, but you entered {P}."
        a = np.zeros((self.N, self.N))
        n_a = int(P * self.N * (self.N - 1) / 2)    # number of connected pairs (edges)

        # Construct list of all pairs of species
        pairs = np.array([(i, j) for i in range(self.N) for j in range(i + 1, self.N)], dtype="i,i")

        # Pick out P links randomly
        sel = np.random.choice(pairs, size=n_a, replace=False)

        # Pick out interactions for each type
        interactions = {}
        for label, p in zip('afcm', [pA, pF, pC, pM]):
            size = int(p * n_a)
            interactions[label] = np.random.choice(sel, size=size, replace=False)
            # get pairs that are in sel but not in interactions
            sel = np.array([pair for pair in sel if pair not in interactions[label]])
        #     remaining_pairs = np.logical_not(np.isin(sel, interactions[label]))
        #     sel = sel[remaining_pairs]
        # # facilitation interaction are set to go either direction: j to i (so a[i,j] non-zero); or i to j (a[j,i] non-zero)
        # for k in range(len(interactions['f'])):
        #     # for half of the facilitation interactions, reverse from i->j to j->i
        #     if np.random.rand() > 0.5:
        #         interactions['f'][k] = (interactions['f'][k][1], interactions['f'][k][0])

        # Capture mutualistic interaction resources
        r_m = {}
        for i, j in interactions['m']:
            if i not in r_m:
                r_m[i] = set()
            if j not in r_m:
                r_m[j] = set()
            r_m[i].add(j)
            r_m[j].add(i)
        r_m = {k: list(v) for k, v in r_m.items()}

        # Resources of facilitators (i.e. who they facilitate)
        r_f = {}
        for i, j in interactions['f']:
            # Facilitations act up the food chain (i facilitates j) #not anymore because the direction of facilitation was randomized
            if j not in r_f:
                r_f[j] = set()
            r_f[j].add(i)  # j facilitates i, non-zero a[i,j]
        r_f = {k: list(v) for k, v in r_f.items()}

        # Resources of competitors (i.e. who they compete with)
        r_c = {}
        for i, j in interactions['c']:
            if i not in r_c:
                r_c[i] = set()
            if j not in r_c:
                r_c[j] = set()
            r_c[i].add(j)
            r_c[j].add(i)
        r_c = {k: list(v) for k, v in r_c.items()}

        # Resources of antagonist predators
        r_a = {}
        for i, j in interactions['a']:
            if j not in r_a:
                r_a[j] = set()
            r_a[j].add(i)
        r_a = {k: list(v) for k, v in r_a.items()}

        # Increase interaction strengths for core species
        def is_core(i, j):
            return (i < core_species or j < core_species)

        # Fill in mutualisms
        for i, j in interactions['m']:
            # (strength of interaction i>j) = (efficiency [random]) * (mutualism strength [random but
            # the same for all mutualistic interactions]) * (proportion of i's mutualistic interactions
            # which are with j)
            factor = nested_factor if is_core(i, j) else 1
            a[i, j] = factor * e[i, j] * self.fM * A[i, j] / np.sum(A[i, r_m[i]])
            a[j, i] = e[j, i] * self.fM * A[j, i] / np.sum(A[j, r_m[j]])

        # Fill in facilitations
        for i, j in interactions['f']:
            # (strength of interaction i>j) = (efficiency [random]) * (facilitation strength [random but
            # the same for all facilitation interactions]) * (proportion of i's facilitation interactions
            # which are with j)interactions

            # Multiply the strength by 2 so that it's equivalent to other interactions where the two directional edges are filled
            # Only j facilitates i. The order of i and j was randomized before
            # the normalization is motivated by a fixed budget of facilitation that the facilitator j can perform
            factor = nested_factor if is_core(i, j) else 1
            a[i, j] = factor * 2 * f[i, j] * self.fF * A[i, j] / np.sum(A[r_f[j], j])
            # only one-directional
            a[j, i] = 0

        # Fill in competitions
        for i, j in interactions['c']:
            # (strength of interaction i>j) = (efficiency [random]) * (competition strength [random but
            # the same for all competition interactions]) * (proportion of i's competition interactions
            # which are with j)
            factor = nested_factor if is_core(i, j) else 1
            a[i, j] = -factor * c[i, j] * self.fC * A[i, j] / np.sum(A[i, r_c[i]])
            a[j, i] = -factor * c[j, i] * self.fC * A[j, i] / np.sum(A[j, r_c[j]])

        # Fill in antagonisms
        for i, j in interactions['a']:
            factor = nested_factor if is_core(i, j) else 1
            a[j, i] = factor * g[j, i] * self.fA * A[j, i] / np.sum(A[j, r_a[j]])  # j (higher index) preys on i --> ensures directionality
            a[i, j] = - a[j, i] / g[j, i]

        # Set diagonal terms to ensure negative intraspecific interactions
        for i in range(self.N):
            a[i, i] = -np.random.uniform()  # Ensuring stability by negative self-regulation

        return a
    
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