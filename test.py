import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.linalg import eigvals
from scipy.optimize import fsolve
import seaborn as sns
import networkx as nx


def initialize_adjacency_matrix(S, E, forbidden_links=None):
    """
    Initialize a random adjacency matrix with S species and E links.
    Ensures no forbidden links are added.

    Parameters:
        S (int): Number of species (nodes).
        E (int): Total number of links (edges).
        forbidden_links (set): Set of forbidden links (tuples of node indices).

    Returns:
        np.ndarray: Initialized adjacency matrix (S x S).
    """
    # Initialize empty adjacency matrix
    adjacency_matrix = np.zeros((S, S), dtype=int)
    links_added = 0

    # Randomly add E links while avoiding forbidden links
    while links_added < E:
        i, j = random.randint(0, S - 1), random.randint(0, S - 1)
        if i != j and adjacency_matrix[i, j] == 0 and (forbidden_links is None or (i, j) not in forbidden_links):
            adjacency_matrix[i, j] = 1
            links_added += 1

    return adjacency_matrix


def swap_links(adjacency_matrix, i, j1, j2, forbidden_links=None):
    """
    Attempt to swap a link in the adjacency matrix.

    Parameters:
        adjacency_matrix (np.ndarray): Current adjacency matrix.
        i (int): Selected row index.
        j1 (int): Index of the current 1 in the column, i.e., A[i, j1] = 1, where A is adjacency matrix.
        j2 (int): Index of the current 0 in the column, i.e., A[i, j2] = 0, where A is adjacency matrix.
        forbidden_links (set): Set of forbidden links (tuples of node indices).

    Returns:
        bool: Whether the swap was performed.
    """
    # Degrees of potential partners
    degree_current_partner = np.sum(adjacency_matrix[:, j1])
    degree_new_partner = np.sum(adjacency_matrix[:, j2])

    # Condition 1: New partner must have a higher degree
    if degree_new_partner <= degree_current_partner:
        return False

    # Condition 2: The swap should not cause the current partner to become extinct
    if np.sum(adjacency_matrix[j1, :]) < 1:  # j1 would have no links after swap
        return False

    # Condition 3: The new link should not be forbidden
    if forbidden_links is not None and (i, j2) in forbidden_links:
        return False

    # Perform the swap
    adjacency_matrix[i, j1] = 0
    adjacency_matrix[i, j2] = 1
    return True


def self_organising_network(S, E, forbidden_links=None, iterations=1000):
    """
    Perform the self-organising network model (SNM).

    Parameters:
        S (int): Number of species (nodes).
        E (int): Total number of links (edges).
        forbidden_links (set): Set of forbidden links (tuples of node indices).
        iterations (int): Number of iterations to perform.

    Returns:
        np.ndarray: Final adjacency matrix.
    """
    # Step 1: Initialize random adjacency matrix
    adjacency_matrix = initialize_adjacency_matrix(S, E, forbidden_links)

    for _ in range(iterations):
        # Step 2.1: Select a random row/column
        i = random.randint(0, S - 1)
        row_or_col = adjacency_matrix[i, :]

        # Get indices of 1s (existing links) and 0s (potential links)
        ones = np.where(row_or_col == 1)[0]
        zeros = np.where(row_or_col == 0)[0]

        if len(ones) == 0 or len(zeros) == 0:
            continue

        # Step 2.2: Pick a random 1 and a random 0
        j1 = random.choice(ones)
        j2 = random.choice(zeros)

        # Step 2.3: Attempt to swap links
        swap_links(adjacency_matrix, i, j1, j2, forbidden_links)

    return adjacency_matrix

def sort_nodes_by_degree(N, interaction_matrix):
    """
    Sort nodes by their degree (number of interactions) in descending order.
    Parameters:
        interaction_matrix: The interaction matrix (adjacency matrix) to analyze.
    Returns:
        Sorted adjacency matrix and the order of nodes.
    """
    # Create a NetworkX graph from the adjacency matrix
    adjacency_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if interaction_matrix[i, j] == 0:
                adjacency_matrix[i, j] = 0
            else:
                adjacency_matrix[i, j] = 1
            if i == j:
                adjacency_matrix[i, j] = 0
    G = nx.from_numpy_array(adjacency_matrix)

    # Compute degrees for each node
    degrees = dict(G.degree())

    # Sort nodes by degree (in descending order)
    sorted_nodes = sorted(degrees, key=degrees.get, reverse=True)

    # Reorder the adjacency matrix
    sorted_matrix = adjacency_matrix[np.ix_(sorted_nodes, sorted_nodes)]

    return sorted_matrix, sorted_nodes

def visualize_adjacency_matrix(N, interaction_matrix):
    """
    Visualizes the adjacency (interaction) matrix as a heatmap and a network graph.
    Parameters:
        interaction_matrix: The interaction matrix (adjacency matrix) to visualize.
    """
    # Sort nodes by degree
    # sorted_matrix, sorted_nodes = sort_nodes_by_degree(N, interaction_matrix)
    sorted_matrix = interaction_matrix
    sorted_nodes = list(range(N))

    # Heatmap of the interaction matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(sorted_matrix, cmap='gray_r', linewidths=0.5, annot=False, square=True)
    # sns.heatmap(sorted_matrix, cmap='coolwarm', linewidths=0.5, annot=False, square=True)
    plt.title("Adjacency Matrix")# Heatmap")# (Sorted by Degree)")
    plt.xlabel("Species")
    plt.ylabel("Species")
    plt.show()

# S = 30  # Number of species
# E = 200  # Number of links
# forbidden_links = {(0, 1), (2, 3)}  # Example forbidden links
#
# adjacency_matrix = self_organising_network(S, E, forbidden_links, iterations=2000)
# print(adjacency_matrix)
#
# visualize_adjacency_matrix(S, adjacency_matrix)

import numpy as np


def sort_adjacency_by_degree(A):
    """
    Sort the adjacency matrix A based on the degrees of the nodes.

    Parameters:
        A (np.ndarray): Adjacency matrix (N x N).

    Returns:
        np.ndarray: Sorted adjacency matrix.
    """
    # Compute degrees (sum of rows + columns for undirected graphs)
    degrees = np.sum(A, axis=0) + np.sum(A, axis=1)

    # Get the sorted indices in descending order
    sorted_indices = np.argsort(degrees)[::-1]

    # Rearrange rows and columns based on sorted indices
    sorted_A = A[np.ix_(sorted_indices, sorted_indices)]

    return sorted_A, sorted_indices


# Example adjacency matrix
A = np.array([
    [0., 0., 0., 1., 1.],
    [0., 0., 0., 1., 1.],
    [0., 0., 0., 1., 0.],
    [1., 1., 1., 0., 1.],
    [1., 1., 0., 1., 0.]
])

# Sort the adjacency matrix
sorted_A, sorted_indices = sort_adjacency_by_degree(A)

print("Original Adjacency Matrix:")
print(A)

print("\nSorted Adjacency Matrix:")
print(sorted_A)

print("\nSorted Indices (Original to Sorted Order):")
print(sorted_indices)

# import matplotlib.pyplot as plt
# import networkx as nx
#
# # Visualize as a graph
# G = nx.from_numpy_array(adjacency_matrix, create_using=nx.DiGraph)
# nx.draw(G, with_labels=True, node_color="lightblue", arrows=True)
# plt.show()
#
# # Visualize as a heatmap
# plt.imshow(adjacency_matrix, cmap="viridis", origin="upper")
# plt.colorbar(label="Link Presence")
# plt.xlabel("Species")
# plt.ylabel("Species")
# plt.title("Adjacency Matrix")
# plt.show()
