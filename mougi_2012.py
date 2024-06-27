import numpy as np
from scipy.optimize import fsolve
from numpy.linalg import eig
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.linalg import solve_continuous_lyapunov


def glv(X, A, r):
    return X * (r + A @ X)

def generate_glv_matrix_cascade(N, fA, fM, fF, fC, pA=0.1, pM=0, pF=0, pC=0):
    """
    Parameters:
        N: number of species
        fA: relative strength of antagonistic interactions
            any positive number?
        fM: relative strength of mutualistic interactions
            any positive number?
        fF: relative facilitation strength
            any pos num?
        PA: density of antagonistic interactions
            float in [0, 1]
        pM: density of mutualistic interactions
            float in [0, 1]
        pF: density of facilitation interactions
            float in [0, 1]
        pC: density of competitive interactions
            float in [0, 1]
    Return:
        a: the actual interaction matrix
            (N x N) matrix
    """
    const_efficiency = 0.5 # to make all efficiencies random as in the paper, use -1 here
    e = np.full((N, N), const_efficiency) if const_efficiency >= 0 else np.random.uniform(0, 1, (N, N))  # mutualistic interaction efficiencies
    g = np.full((N, N), const_efficiency) if const_efficiency >= 0 else np.random.uniform(0, 1, (N, N))  # trophic interaction efficiencies
    f = np.full((N, N), const_efficiency) if const_efficiency >= 0 else np.random.uniform(0, 1, (N, N))  # facilitation interaction efficiencies
    c = np.full((N, N), const_efficiency) if const_efficiency >= 0 else np.random.uniform(0, 1, (N, N))  # competitive interaction efficiencies
    A = np.random.uniform(0, 1, (N, N))  # potential preference for the interaction partners
    np.fill_diagonal(A, 0)

    P = pA + pF + pC + pM
    assert P <= 1, "Total interaction density must be less than 1, but you entered {} + {} + {} + {} = {}".format(
        pA, pM, pF, pC, P
    )

    a = np.zeros((N, N))
    n_a = int(P * N * (N - 1) / 2)  # number of connected pairs (edges) [SH]

    # Construct list of all pairs of species
    pairs = np.array([(i, j) for i in range(N) for j in range(i + 1, N)], dtype="i,i")

    # Pick out P links randomly
    sel = np.random.choice(pairs, size=n_a, replace=False)

    # Pick out interactions for each type
    interactions = {}
    for label, p in zip('afcm', [pA, pF, pC, pM]):
        size = int(p * n_a)
        interactions[label] = np.random.choice(sel, size=size, replace=False)
        # get pairs that are in sel but not in interactions
        remainingpairs = np.logical_not(np.isin(sel, interactions[label]))
        sel = sel[remainingpairs]
    # facilitation interaction are set to go either direction: j to i (so a[i,j] non-zero); or i to j (a[j,i] non-zero)
    for k in range(len(interactions['f'])):
        # for half of the facilitation interactions, reverse from i->j to j->i 
        if np.random.rand() > 0.5:
            interactions['f'][k] = (interactions['f'][k][1], interactions['f'][k][0])

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

    # To make the facilitations random, randomly shuffle each (i, j)
    # pair in the list [(i, j) ... ] of facilitations here.
    np.random.shuffle(interactions['f']) # why is this necessary?

    # Resources of facilitators (i.e. who they facilitate)
    r_f = {}
    for i, j in interactions['f']:
        # Facilitations act up the food chain (i facilitates j) #not anymore because the direction of facilitation was randomized
        if j not in r_f:
            r_f[j] = set()
        r_f[j].add(i) # j facilitates i, non-zero a[i,j]
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

    # Fill in mutualisms
    for i, j in interactions['m']:
        # (strength of interaction i>j) = (efficiency [random]) * (mutualism strength [random but
        # the same for all mutualistic interactions]) * (proportion of i's mutualistic interactions
        # which are with j)
        a[i, j] = e[i, j] * fM * A[i, j] / np.sum(A[i, r_m[i]])
        a[j, i] = e[j, i] * fM * A[j, i] / np.sum(A[j, r_m[j]])

    # Fill in facilitations
    for i, j in interactions['f']:
        # (strength of interaction i>j) = (efficiency [random]) * (facilitation strength [random but
        # the same for all facilitation interactions]) * (proportion of i's facilitation interactions
        # which are with j)interactions

        # Multiply the strength by 2 so that it's equivalent to other interactions where the two directional edges are filled
        # Only j facilitates i. The order of i and j was randomized before
        # the normalization is motivated by a fixed budget of facilitation that the facilitator j can perform
        a[i, j] = 2 * f[i, j] * fF * A[i, j] / np.sum(A[r_f[j], j])
        # only one-directional
        a[j, i] = 0
            

    # Fill in competitions
    for i, j in interactions['c']:
        # (strength of interaction i>j) = (efficiency [random]) * (competition strength [random but
        # the same for all competition interactions]) * (proportion of i's competition interactions
        # which are with j)
        a[i, j] = - c[i, j] * fC * A[i, j] / np.sum(A[i, r_c[i]])
        a[j, i] = - c[j, i] * fC * A[j, i] / np.sum(A[j, r_c[j]])

    # Fill in antagonisms
    for i, j in interactions['a']:
        a[j, i] = g[j, i] * fA * A[j, i] / np.sum(A[j, r_a[j]]) # j (higher index) preys on i --> ensures directionality
        a[i, j] = - a[j, i] / g[j, i]

    # Fill in s_i values (i.e. a_ii), which must be negative
    for i in range(N):
        a[i, i] = - np.random.uniform()

    return a


def generate_lv_params(N, pA=0.1, pM=0, pF=0, pC=0):
    """
    Generate the parameters to construct the basic GLV interaction matrix.
    """

    fA = 1
    fM = 1
    fF = 1
    fC = 1

    a = generate_glv_matrix_cascade(N, fA, fM, fF, fC, pA, pM, pF, pC)

    X_eq = np.random.uniform(0, 1, N)

    # Pick r such that X_eq is an equilibrium
    def glv_r(r):
        return X_eq * (r + a @ X_eq)

    r0 = np.ones(N) / 2  # Initial guess for fsolve
    r = fsolve(glv_r, r0)

    return r, a, X_eq

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
N = 50  # Number of species
ntrial = 300 # number of sampled "models" for a given set of parameter

stabilities = {}
for P in tqdm([0.1, 0.3, 0.5, 0.7, 0.9]):
    for PM in np.arange(0, 1, 0.05):
        pM = P*PM
        pA = P*(1-PM)
        s = []
        for trial in range(300):
            r, a, X_eq = generate_lv_params(N, pA, pM, 0.,0.)
            #print(r)

            stable = stability(a, X_eq)

            s += [stable]
        stabilities[(P, PM)] = np.mean(s)

import pandas as pd

print(pd.Series(stabilities))
df = pd.Series(stabilities).unstack().T
print(df)


df.plot(marker='o', cmap='cool', mec='k')
handles, labels = plt.gca().get_legend_handles_labels()
order = list(range(len(labels)))[::-1]
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],
          title = "P", loc="upper left", bbox_to_anchor=(1,1))
plt.xlabel("$p_M$ (mutualism)")
plt.ylabel("Fraction stable equilibria")
plt.title(f"N={N}", fontsize=14)
plt.savefig(f"mougi_fig1rep_N{N}.png", bbox_inches="tight")
