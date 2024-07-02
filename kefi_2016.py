import numpy as np
from scipy.optimize import fsolve
from numpy.linalg import eig
import matplotlib.pyplot as plt
from tqdm import tqdm
from os import system

def randomize_interaction_matrix(adj, type, strength=1.0):
    """
    This function takes an adjacency matrix to randomize the entries a_ij, which represent
    interaction strengths between species i and j.

    :param
        adj (array): adjacency matrix with entries 0 and 1 only
        type (str): Interaction types ('a': antagonism = trophic interaction;
                    'f': facilitation = non-trophic positive interaction;
                    'c': competition = non-trophic negative interaction)
        strength (float): interaction strength
    :return:
        a_rand: randomized interaction matrix
    """
    N = len(adj)
    a_rand = np.zeros((N, N))

    const_efficiency = 0.85  # Fix all efficiencies as in S2Table.DOCX
    e = np.full((N, N), const_efficiency)
    A = np.random.uniform(0, 1, (N, N))  # potential preference for the interaction partners
    np.fill_diagonal(A, 0)

    count_links = 0
    interactions = []
    for i in range(adj.shape[0]):
        for j in range(adj.shape[1]):
            if adj[i, j] == 1:
                interactions.append([i, j])
                count_links += 1

    monodirectional = []
    bidirectional = []
    bidirectional_twin = []
    canibalism = []
    for i, j in interactions:
        if i == j:
            canibalism.append([i, j])
        else:
            if [j, i] in interactions and j > i:
                bidirectional.append([i, j])
                bidirectional_twin.append([j, i])
        if [i, j] not in canibalism and [i, j] not in bidirectional and [j, i] not in bidirectional_twin:
            if j > i:
                monodirectional.append([i, j])

    n = 4623    # num of links in the Chilean data
    num_interactions = 0
    proportion = 0.

    if type == 'a':     # Tropic interactions
        fA = strength
        interactions = monodirectional
        # Resources of antagonist predators
        r_a = {}
        for i, j in interactions:
            if j not in r_a:
                r_a[j] = set()
            r_a[j].add(i)
        r_a = {k: list(v) for k, v in r_a.items()}
        # Fill in antagonisms
        for i, j in interactions:
            a_rand[j, i] = e[j, i] * fA * A[j, i] / np.sum(A[j, r_a[j]])  # j (higher index) preys on i --> ensures directionality
            a_rand[i, j] = - a_rand[j, i] / e[j, i]
        num_interactions = len(interactions)
        proportion = num_interactions / n

    elif type == 'f':   # Facilitating interactions
        fF = strength
        interactions = monodirectional
        # Resources of facilitators (i.e. who they facilitate)
        r_f = {}
        for i, j in interactions:
            # Facilitations act up the food chain (i facilitates j) #not anymore because the direction of facilitation was randomized
            if j not in r_f:
                r_f[j] = set()
            r_f[j].add(i)  # j facilitates i, non-zero a[i,j]
        r_f = {k: list(v) for k, v in r_f.items()}
        for i, j in interactions:
            # (strength of interaction i>j) = (efficiency [random]) * (facilitation strength [random but
            # the same for all facilitation interactions]) * (proportion of i's facilitation interactions
            # which are with j)interactions

            # Multiply the strength by 2 so that it's equivalent to other interactions where the two directional edges are filled
            # Only j facilitates i. The order of i and j was randomized before
            # the normalization is motivated by a fixed budget of facilitation that the facilitator j can perform
            a_rand[i, j] = 2 * e[i, j] * fF * A[i, j] / np.sum(A[r_f[j], j])
            # only one-directional
            a_rand[j, i] = 0
        num_interactions = len(interactions)
        proportion = num_interactions / n

    elif type == "m":
        fM = strength
        interactions = bidirectional
        # Resources of mutualists (i.e., bidirectional benefits)
        r_m = {}
        for i, j in interactions:
            if i not in r_m:
                r_m[i] = set()
            if j not in r_m:
                r_m[j] = set()
            r_m[i].add(j)
            r_m[j].add(i)
        r_m = {k: list(v) for k, v in r_m.items()}
        # Fill in mutualisms
        for i, j in interactions:
            # (strength of interaction i>j) = (efficiency [random]) * (mutualism strength [random but
            # the same for all mutualistic interactions]) * (proportion of i's mutualistic interactions
            # which are with j)
            a_rand[i, j] = e[i, j] * fM * A[i, j] / np.sum(A[i, r_m[i]])
            a_rand[j, i] = e[j, i] * fM * A[j, i] / np.sum(A[j, r_m[j]])
        num_interactions = len(interactions)
        proportion = num_interactions / n

    elif type == 'c':   # Competitive interactions
        fC = strength
        interactions = bidirectional + monodirectional
        # Resources of competitors (i.e. who they compete with)
        r_c = {}
        for i, j in interactions:
            if i not in r_c:
                r_c[i] = set()
            if j not in r_c:
                r_c[j] = set()
            r_c[i].add(j)
            r_c[j].add(i)
        r_c = {k: list(v) for k, v in r_c.items()}
        # Fill in competitions
        for i, j in interactions:
            # (strength of interaction i>j) = (efficiency [random]) * (competition strength [random but
            # the same for all competition interactions]) * (proportion of i's competition interactions
            # which are with j)
            a_rand[i, j] = - e[i, j] * fC * A[i, j] / np.sum(A[i, r_c[i]])
            a_rand[j, i] = - e[j, i] * fC * A[j, i] / np.sum(A[j, r_c[j]])
        num_interactions = len(interactions)
        proportion = num_interactions / n
    print(num_interactions, f"links lead to p = {proportion:.4f}")

    # Fill in s_i values (i.e. a_ii), which must be negative
    for i in range(N):
        a_rand[i, i] = - np.random.uniform()

    return a_rand


def generate_glv_params(N, interaction_matrix):
    """
    Generate the parameters to construct the basic GLV interaction matrix.
    """
    X_eq = np.random.uniform(0, 1, N)

    # Pick r such that X_eq is an equilibrium
    def glv_r(r):
        return X_eq * (r + interaction_matrix @ X_eq)

    r0 = np.ones(N) / 2  # Initial guess for fsolve
    r = fsolve(glv_r, r0)

    return r, interaction_matrix, X_eq


def stability(a, X_eq):
    # Calculate M, the community matrix, i.e. the Jacobian at X_eq
    # For the GLV equations this is very simple
    # M_ij = X_eq_i & a_ij
    M = X_eq * a
    eigenvalues, _ = eig(M)
    return all(np.real(eigenvalues) < 0)


data_path = "chilean_data"
interaction_type = {"a": "TI", "f": "NTIpos", "m": "NTIpos", "c": "NTIneg"}

system(f"mkdir {data_path}/output/stable {data_path}/output/unstable")
num_species = 106
num_trials = 300
count = 0
for run in range(num_trials):
    multi_interaction = np.zeros((num_species, num_species))
    for interaction_code in list(interaction_type.keys()):
        # print(f"({interaction_code})", interaction_type[interaction_code])
        file_name = f"chilean_{interaction_type[interaction_code]}"
        f = open(f"{data_path}/{file_name}.txt", "r")
        adj = []
        for line in f.readlines()[1:]:
            line = line.strip().split('\t')
            adj.append([int(elt) for elt in line[2:]])
        adj = np.reshape(adj, (len(adj), len(adj)))
        a_rand = randomize_interaction_matrix(adj, type=interaction_code)
        multi_interaction += a_rand
    r, a, X_eq = generate_glv_params(num_species, multi_interaction)
    stable = stability(a, X_eq)
    if stable:
        count += 1
        np.savetxt(f"{data_path}/output/stable/chilean_multiinteraction_rand{run}.txt", multi_interaction, fmt="%f")
    else:
        np.savetxt(f"{data_path}/output/unstable/chilean_multiinteraction_rand{run}.txt", multi_interaction, fmt="%f")
print(f"{count} out of {num_trials} networks are stable!")


### To do : Visualization of the stability assessment result!!

