import numpy as np
from scipy.optimize import fsolve
from numpy.linalg import eig
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.linalg import solve_continuous_lyapunov


def glv(X, A, r):
    return X * (r + A @ X)


# def glv_holland_ii(X, r, n, h, fa, fm, A, mutualisms, antagonisms):
#     """
#     Calculate the population derivatives for a Holland type II functional response.
#     """
#     a = generate_holland_matrix_cascade(X, n, h, fa, fm, A, mutualisms, antagonisms)
#     return X * (r + A @ X)


def generate_glv_matrix_cascade(N, fA, fM, fF, P=0.1, pM=0, pF=0):
    """
    Parameters:
        N: number of species
        fA: relative strength of antagonistic interactions
            any positive number?
        fM: relative strength of mutualistic interactions
            any positive number?
        fF: relative facilitation strength
            any pos num?
        P: density of interaction matrix (= proportion of connected pairs [SH])
            float in [0, 1]
        pM: probability of mutualistic interactions, within those P interactions
            float in [0, 1]
        pF: probability of facilitation interactions, within those P interactions
            float in [0, 1]
    Return:
        a: the actual interaction matrix
            (N x N) matrix
    """
    e = np.random.uniform(0, 1, (N, N))  # mutualistic interaction efficiencies
    g = np.random.uniform(0, 1, (N, N))  # trophic interaction efficiencies
    f = np.random.uniform(0, 1, (N, N))  # facilitation interaction efficiencies
    A = np.random.uniform(0, 1, (N, N))  # potential preference for the interaction partners
    np.fill_diagonal(A, 0)

    a = np.zeros((N, N))
    n_a = int(P * N * (N - 1) / 2)  # number of connected pairs (edges) [SH]

    # Construct list of all pairs of species
    pairs = np.array([(i, j) for i in range(N) for j in range(i + 1, N)])

    # Pick out P antagonistic links randomly
    sel = np.random.choice(len(pairs), size=n_a, replace=False)
    antagonisms = pairs[sel]
    np.savetxt("antagonism.txt", antagonisms, fmt="%i")

    # Pick out mutualisms and facilitations from antagonisms
    n_alt = int((pM + pF) * n_a)
    alt_interactions = np.random.choice(len(antagonisms), size=n_alt, replace=False)
    mutual_interactions = alt_interactions[:int(pM * n_a)]
    facilitation_interactions = alt_interactions[int(pM * n_a):]
    mutualisms = antagonisms[mutual_interactions]
    facilitations = antagonisms[facilitation_interactions]
    np.savetxt("mutualisms.txt", mutualisms, fmt="%i")
    np.savetxt("facilitations.txt", facilitations, fmt="%i")

    # Resources of mutualists
    r_m = {}
    for i, j in mutualisms:
        if i not in r_m:
            r_m[i] = set()
        if j not in r_m:
            r_m[j] = set()
        r_m[i].add(j)
        r_m[j].add(i)
    r_m = {k: list(v) for k, v in r_m.items()}

    # To make the facilitations random, randomly shuffle each (i, j)
    # pair in the list [(i, j) ... ] of facilitations here.

    # Resources of facilitators (i.e. who they facilitate)
    r_f = {}
    for i, j in facilitations:
        # Facilitations act up the food chain (i facilitates j)
        if i not in r_f:
            r_f[i] = set()
        r_f[i].add(j)
    r_f = {k: list(v) for k, v in r_f.items()}

    # Remove selected mutualisms and facilitations from antagonisms
    mask = np.ones(len(antagonisms), dtype=bool)
    mask[mutual_interactions] = False
    mask[facilitation_interactions] = False
    antagonisms = antagonisms[mask]
    np.savetxt("antagonism_mask.txt", antagonisms, fmt="%i")

    """
    Here is where you would select e.g. pF facilitation interactions
    or pC competitive interactions
    """

    # Resources of antagonist predators
    r_a = {}
    for i, j in antagonisms:
        if j not in r_a:
            r_a[j] = set()
        r_a[j].add(i)
    r_a = {k: list(v) for k, v in r_a.items()}

    # Fill in mutualisms
    for i, j in mutualisms:
        # (strength of interaction i>j) = (efficiency [random]) * (mutualism strength [random but
        # the same for all mutualistic interactions]) * (proportion of i's mutualistic interactions
        # which are with j)
        a[i, j] = e[i, j] * fM * A[i, j] / np.sum(A[i, r_m[i]])
        a[j, i] = e[j, i] * fM * A[j, i] / np.sum(A[j, r_m[j]])

    # Fill in facilitations
    for i, j in facilitations:
        # (strength of interaction i>j) = (efficiency [random]) * (facilitation strength [random but
        # the same for all facilitation interactions]) * (proportion of i's facilitation interactions
        # which are with j)
        a[i, j] = 0
        a[j, i] = f[j, i] * fF * A[i, j] / np.sum(A[i, r_f[i]])


    # Fill in antagonisms
    for i, j in antagonisms:
        a[j, i] = g[j, i] * fA * A[j, i] / np.sum(A[j, r_a[j]]) # j (higher index) preys on i --> ensures directionality
        a[i, j] = - a[j, i] / g[j, i]

    # Fill in s_i values (i.e. a_ii), which must be negative
    for i in range(N):
        a[i, i] = - np.random.uniform()

    return a


def generate_lv_params(N, P, pM=0, pF=0):
    """
    Generate the parameters to construct the basic GLV interaction matrix.
    """

    fA = 2
    fM = 1
    fF = 1

    a = generate_glv_matrix_cascade(N, fA, fM, fF, P, pM, pF)

    X_eq = np.random.uniform(0, 1, N)

    # Pick r such that X_eq is an equilibrium
    def glv_r(r):
        return X_eq * (r + a @ X_eq)

    r0 = np.ones(N) / 2  # Initial guess for fsolve
    r = fsolve(glv_r, r0)

    return r, a, X_eq


def generate_holling_params(N, P, pM):
    pass


def generate_holling_matrix_cascade(X, n, h, fa, fm, A, mutualisms, antagonisms):
    pass


def stability(a, X_eq):
    # Calculate M, the community matrix, i.e. the Jacobian at X_eq
    # For the GLV equations this is very simple
    # M_ij = X_eq_i & a_ij
    M = X_eq * a
    eigenvalues, _ = eig(M)
    return all(np.real(eigenvalues) < 0)

def global_stability(a):
    # Test whether the interaction matrix is Lyapunov-diagonally stable,
    # which means the system has a global stable equilibrium.
    Q = np.eye(len(a))  # Identity matrix as positive definite matrix

    # Solve the Lyapunov equation A.T * D + D * A = -Q
    D = solve_continuous_lyapunov(a.T, -Q)
    
    # Check if D is a positive diagonal matrix
    is_diagonal = np.allclose(D, np.diag(np.diagonal(D)))
    is_positive_diagonal = np.all(np.diagonal(D) > 0)
    
    return is_diagonal and is_positive_diagonal

# Example usage
N = 20  # Number of species

stabilities = {}
for P in tqdm([0.2, 0.4, 0.5, 0.6, 0.8]):
    for pM in np.arange(0, 1, 0.05):
        s = []
        for trial in range(100):
            r, a, X_eq = generate_lv_params(N, P, pM)

            stable = stability(a, X_eq)

            s += [stable]
        stabilities[(P, pM)] = np.mean(s)

import pandas as pd

df = pd.Series(stabilities).unstack().T


df.plot(marker='o', cmap='cool', mec='k')
handles, labels = plt.gca().get_legend_handles_labels()
order = list(range(len(labels)))[::-1]
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],
          title = "P", loc="upper left", bbox_to_anchor=(1,1))
plt.xlabel("$p_M$ (mutualism)")
plt.ylabel("Fraction stable equilibria")
plt.title(f"N={N}", fontsize=14)
plt.savefig(f"mougi_fig1rep_N{N}.png", bbox_inches="tight")
