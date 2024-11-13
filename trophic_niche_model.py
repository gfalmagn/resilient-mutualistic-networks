import numpy as np
from scipy.optimize import fsolve
import scipy
import matplotlib.pyplot as plt
import networkx as nex
import random
import pandas as pan

def generate_unique_uniform_floats(N, low, high):
    """
    Generate N unique uniformly-distributed floats between low and high using NumPy, through rejection sampling.
    """
    float_N = np.random.uniform(low=low, high=high, size=10*N)
    return np.unique(float_N)[:N]

def trophic_levels(A):
    """
    Calculate the trophic levels of species in a network with adjacency matrix A using Levine 1980 algorithm.
    Inputs:
        A = Weighted Adjacency Matrix for Trophic Interactions (Array of shape S x S)
    Outputs:
        trophic_levels = Trophic levels of each species (Array of shape S)
    """
    G = nex.from_numpy_array(A, create_using=nex.DiGraph)
    # Convert dictionary of nodes and their trophic levels to a np array
    return np.array(list(nex.trophic_levels(G).values()))


def generate_classic_niche_model(S, pA, pC, competition_type='indirect'):
    """
    Creates a network of S species using the classic niche model (Williams & Martinez 2000) with the given parameters.
    Inputs 
        S = number of species
        pA= density of antagonistic interactions (as a fraction of all possible interactions = S^2)
            float in [0, 1]
        pC: density of competitive interactions
            float in [0, 1]
        competition_type: 'indirect' or 'direct' (default: 'indirect')
            'indirect' - competition is mediated through shared resources, and is modelled indirectly as predator interference
            affecting the weighted adjacency matrix A (lower values of A[i,j] indicate stronger competition b/w species i and other predators of prey species j)
            'direct' - competition is direct, and is modelled through a direct competition matrix D, higher values of D[i,j] indicate stronger competition b/w species i and species j
            with weighted adjacency matrix A unaffected 

    Outputs:
        A = Weighted Adjacency Matrix for Trophic Interactions (Array of shape S x S)
        D = Predator Interference Competition Matrix (Array of shape S x S)
        niche_values = niche values for each species (Array of shape S)
        niche_centers = niche centers for each species (Array of shape S)
        niche_breadths = niche breadths for each species (Array of shape S)

    NOTE:
    I) Pass pC = 0  to generate ONLY a classic trophic (predator-prey) network
    II) This model ASSUMES EQUAL PREFERENCE FOR ALL PREY that fall within the niche breadth of a predator

    """

    L = pA * S**2

    # 1. Get the niche value n_i for each species
    niche_values = np.sort(generate_unique_uniform_floats(S, 0, 1))
    if len(niche_values) < S:
        print(f"Warning: only {len(niche_values)} unique niche values generated for {S} species.")
        print(" Regenerating niche values...")
        return generate_classic_niche_model(S, pA, pC)

    # 2. Define the predation breadth r_i
    # First the alpha and beta values picked so that the mean of the beta distribution is 2C
    # 1 / 1 + b = 2C --> b = (1 - 2C) / 2C
    b = (1 - 2 * pA) / (2 * pA)

    if b < 0:
        print(f"Warning: b = {b} < 0; Given pA = {pA} MUST be less than 0.5 for the model to work.")
        return None, None, None, None, None

    niche_breadths = niche_values * 2 * np.random.beta(1, b, size=S)

    # Pick the niche centers
    niche_centers = np.random.uniform(low=niche_breadths/2, high=niche_values, size=S)

    # Create the unweighted adjacency matrix (TR) using notation established in Kefi 2016.
    # TR is (S x S) matrix where TR[i, j] = 1 if species i preys on species j, 0 otherwise.
    TR = np.zeros((S, S), dtype=int)
    for i in range(1, S): # Start at 1 so that species 0 is default autotroph
        TR[i] = ((niche_values < niche_centers[i] + niche_breadths[i] / 2) & 
                (niche_values > niche_centers[i] - niche_breadths[i] / 2)).astype(np.int64)
    

    G = nex.from_numpy_array(TR, create_using=nex.DiGraph)
    cycles = nex.simple_cycles(G)
    try:    
        cycle = next(cycles)
    except StopIteration:
        cycle = []
    found_cycle = (len(cycle) > 1)
    if found_cycle:
        print(f"found cycle {cycle}; recalculating...")
    if (not nex.is_connected(G.to_undirected())) or found_cycle:
        # If the network is not connected, or if there is a cycle, then we need to re-run the model
        return generate_classic_niche_model(S, pA, pC)

    # Next, create the weighted adjacency matrix A and the predator interference competition matrix D, 
    # based on the unweighted adjacency matrix TR and values of pC.
    if pC == 0:
        D = np.zeros((S, S)) # No competition
        # Create a different preference matrix (B) of size S x S

        ''' ASSUMING EQUAL PREFERENCE FOR ALL SPECIES 
        i.e. B[i,j] = 1/[# of Prey of i] for all j such that TR[i,j] =1
        '''
        B = np.zeros((S, S), dtype=int)
        for i in range(S):
            B[i] = TR[i]/np.sum(TR[i]) 
            # Preference of species i for each species j, such that sum(B[i]) = 1 [Normalised prefrences]

        # Create the weighted adjacency matrix A
        A = B
        return A, D, niche_values, niche_centers, niche_breadths
    else:
        # Create the predator interference competition matrix D
        # Vectorized computation of the predator interference competition matrix D
        niche_centers_matrix = np.tile(niche_centers, (S, 1)) # S x S matrix where each row represents the niche centers of all species
        niche_breadths_matrix = np.tile(niche_breadths, (S, 1)) # S x S matrix where each row represents the niche breadths of all species

        overlap_matrix = (np.minimum(niche_centers_matrix + niche_breadths_matrix / 2, niche_centers_matrix.T + niche_breadths_matrix.T / 2) - 
                  np.maximum(niche_centers_matrix - niche_breadths_matrix / 2, niche_centers_matrix.T - niche_breadths_matrix.T / 2)) / niche_breadths_matrix.T
        ''' NOTE: COMPETITION IS NOT SYMMETRIC, i.e. D[i,j] != D[j,i] in general, i.e. 
        a specialist species i will generally more affected by competition from generalist species j than vice versa.
        Check notes for more details.
        '''
        # Values less than 0 indicate no overlap, and are set to 0
        D = np.maximum(overlap_matrix, 0)

        # Next, check if the number of competitive interactions is less than pC * S^2. 
        # If not, only the pC * S^2 strongest competitive interactions are retained.
        if np.sum(D > 0) > pC * S**2:
            # Get the pC * S^2 largest values of D
            D_flat = D.flatten()
            D_flat.sort()
            threshold = D_flat[-int(pC * S**2)]
            D[D < threshold] = 0
        

        # Create the preference matrix B, based on whether competition is direct or indirect
        if competition_type == 'direct':
            # Here A is unaffected by competition.
            ''' ASSUMING EQUAL PREFERENCE FOR ALL SPECIES
            i.e. B[i,j] = 1/[# of Prey of i] for all j such that TR[i,j] =1
            '''
            B = np.zeros((S, S), dtype=int)
            for i in range(S):
                B[i] = TR[i]/np.sum(TR[i]) 
            # Preference of species i for each species j, such that sum(B[i]) = 1 [Normalised prefrences]
            assert np.allclose(np.sum(B, axis=1), 1), "WARNING: reference matrix B is not normalised!"
            A = B
        elif competition_type == 'indirect':
            B = np.zeros((S, S))
            # Here A is affected by competition.
            ''' ASSUMING EQUAL PREFERENCE FOR ALL SPECIES:
            B[i,j] = TR[i,j]/[ SUM_l (D[l, i]* TR[l, j]) + (# of Prey of i) ] for all j such that TR[i,j] = 1
            '''
            ''' UNVECTORIZED IMPLEMENTATION
            for i in range(S):
                for j in range(S):
                    if TR[i, j] == 1:
                        B[i, j] = TR[i, j] / (np.sum(D[:, i] * TR[:, j]) + np.sum(TR[i]))
            '''

            # Vectorized computation of the preference matrix B
            prey_counts = np.sum(TR, axis=1)  # Row vector with number of prey for each species i (Array of shape S)
            competition_effects = D.T @ TR  # Competition effects on each prey species
            B = TR / (competition_effects + prey_counts[:, np.newaxis])
            B[TR == 0] = 0  # Ensure B[i, j] is 0 where TR[i, j] is 0
            A = B
        else:
            print("Invalid competition type; choose 'direct' or 'indirect'")
            return None, None, None, None, None
        
        return A, D, niche_values, niche_centers, niche_breadths



def generate_adapted_niche_model(S, pA, pC, competition_type='indirect'):
    """
    Creates a network of S species using the adapted niche model (Alhoff 2016) with the given parameters.
    Inputs 
        S = number of species
        pA= density of antagonistic interactions (as a fraction of all possible interactions = S^2)
            float in [0, 1]
        pC: density of competitive interactions
            float in [0, 1]
        competition_type: 'indirect' or 'direct' (default: 'indirect')
            'indirect' - competition is mediated through shared resources, and is modelled indirectly as predator interference
            affecting the weighted adjacency matrix A (lower values of A[i,j] indicate stronger competition b/w species i and other predators of prey species j)
            'direct' - competition is direct, and is modelled through a direct competition matrix D, higher values of D[i,j] indicate stronger competition b/w species i and species j
            with weighted adjacency matrix A unaffected 

    Outputs:
        A = Weighted Adjacency Matrix for Trophic Interactions (Array of shape S x S)
        D = Predator Interference Competition Matrix (Array of shape S x S)
        niche_values = niche values for each species (Array of shape S)
        niche_centers = niche centers for each species (Array of shape S)
        niche_breadths = niche breadths for each species (Array of shape S)

    NOTE:
    I) Pass pC = 0  to generate ONLY a classic trophic (predator-prey) network
    II) This model ASSUMES GAUSSIAN PREFERENCE FOR PREY that fall within the niche breadth of a predator

    """
    L = pA * S**2 # Number of antagonistic interactions

    # 1. Get the niche value n_i for each species
    niche_values = np.sort(generate_unique_uniform_floats(S, 0, 1))
    if len(niche_values) < S:
        print(f"Warning: only {len(niche_values)} unique niche values generated for {S} species.")
        print(" Regenerating niche values...")
        return generate_adapted_niche_model(S, pA, pC)

    # 2. Define the predation breadth r_i
    # First the alpha and beta values picked so that the mean of the beta distribution is 2C
    # 1 / 1 + b = 2C --> b = (1 - 2C) / 2C
    b = (1 - 2 * pA) / (2 * pA)

    if b < 0:
        print(f"Warning: b = {b} < 0; Given pA = {pA} MUST be less than 0.5 for the model to work.")
        return None, None, None, None, None

    niche_breadths = niche_values * 2 * np.random.beta(1, b, size=S)

    # Pick the niche centers
    niche_centers = np.random.uniform(low=niche_breadths/2, high=niche_values, size=S)

    # Create the unweighted adjacency matrix (TR) using notation established in Kefi 2016.
    # TR is (S x S) matrix where TR[i, j] = 1 if species i preys on species j, 0 otherwise.
    TR = np.zeros((S, S), dtype=int)
    for i in range(1, S): # Start at 1 so that species 0 is default autotroph
        TR[i] = ((niche_values < niche_centers[i] + niche_breadths[i] / 2) & 
                (niche_values > niche_centers[i] - niche_breadths[i] / 2)).astype(np.int64)
    

    G = nex.from_numpy_array(TR, create_using=nex.DiGraph)
    cycles = nex.simple_cycles(G)
    try:    
        cycle = next(cycles)
    except StopIteration:
        cycle = []
    found_cycle = (len(cycle) > 1)
    if found_cycle:
        print(f"found cycle {cycle}; recalculating...")
    if (not nex.is_connected(G.to_undirected())) or found_cycle:
        # If the network is not connected, or if there is a cycle, then we need to re-run the model
        return generate_adapted_niche_model(S, pA, pC)

    
        

