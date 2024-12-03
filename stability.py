from EcoResilience import GLVmodel
import numpy as np
from os import system

""" Stability assessment of nested model """
# # Nested model parameters
# N = 50  # Number of species
# num_tests = 300
# p_a, p_m, p_f, p_c = 0.4, 0.4, 0.1, 0.1  # Interaction type proportions
# nestedness_level = 0.7  # Proportion of core species
# nested_factor = 1.
# model = GLVmodel(num_species=N)
# system(f"mkdir nested/pa{p_a}_pm{p_m}_pf{p_f}_pc{p_c}")
#
# save = True
# stability = []
# for run in range(num_tests):
#     # Generate parameters and assess stability
#     r, interaction_matrix, X_eq = model.generate_glv_params(p_a, p_m, p_f, p_c, nestedness_level, nested_factor)
#     # print(interaction_matrix)
#     stable = model.check_stability(interaction_matrix, X_eq)
#     print(run, stable)
#     if stable:
#         np.savetxt(f"nested/pa{p_a}_pm{p_m}_pf{p_f}_pc{p_c}/N{N}_nl{nestedness_level}_nf{nested_factor:.1f}_Run{run}.txt",
#                    interaction_matrix, fmt="%.8f")
#
#     stability.append(stable)
#
#     # model.visualize_adjacency_matrix(interaction_matrix)
#     # print("Equilibrium abundances:", X_eq)
# print(f"{sum(stability)}/{num_tests} networks (N={N}) are stable !")


""" Application of nestedness to the Chilean data from Kefi et al. (PLOS Biology, 2016) """
data_path = "chilean_data"
interaction_type = {"a": "TI", "f": "NTIpos", "m": "NTIpos", "c": "NTIneg"}
# 'a': antagonism = trophic interaction;
# 'f': facilitation = non-trophic positive interaction;
# 'c': competition = non-trophic negative interaction
num_species = 106
num_links = 4303  # num of links of every interaction type in the Chilean data
model = GLVmodel(num_species)

# Extract the proportion of each interaction type
p = {}
for interaction_code in list(interaction_type.keys()):
    file_name = f"chilean_{interaction_type[interaction_code]}"
    f = open(f"{data_path}/{file_name}.txt", "r")
    adj = []
    for line in f.readlines()[1:]:
        line = line.strip().split('\t')
        adj.append([int(elt) for elt in line[2:]])
    adj = np.array(adj)
    adj = np.reshape(adj, (len(adj), len(adj)))
    proportion = model.extract_proportion_of_interactions(adj, num_total_links=num_links, interaction_type=interaction_code)
    # print(f"p_{interaction_code} = {proportion}")
    p[interaction_code] = proportion

    # OUTPUT
    # 231 links lead to p_a = 0.0500
    # 110 links lead to p_f = 0.0238
    # 6 links lead to p_m = 0.0013
    # 1540 links lead to p_c = 0.3331


num_tests = 10
# level = 0.7
for level in np.arange(0, 1, 0.1):
    print(f"Nestedness level = {level}")
    stability = []
    count = 0
    for run in range(num_tests):
        # Generate a matrix with random interaction strengths
        r, interaction_matrix, X_eq = model.generate_glv_params(p["a"], p["m"], p["f"], p["c"], nestedness_level=level, nested_factor=1.0)
        stable = model.check_stability(interaction_matrix, X_eq)
        if stable:
            count += 1
            print(count, stable)
            np.savetxt(f"nested/chilean/nl{level:.1f}_Run{run}.txt", interaction_matrix, fmt="%.8f")

        stability.append(stable)

        # model.visualize_adjacency_matrix(interaction_matrix)
        # print("Equilibrium abundances:", X_eq)
    print(f"{sum(stability)}/{num_tests} networks (N={num_species}) are stable !")



