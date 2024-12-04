from EcoResilience import GLVmodel
import numpy as np
from os import system
import matplotlib.pyplot as plt
from tqdm import tqdm

""" Stability assessment of nested model """
# # Nested model parameters
# N = 50  # Number of species
# num_tests = 300
# p_a, p_m, p_f, p_c = 0.4, 0.4, 0.1, 0.1  # Interaction type fractions
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
interaction_type = {"a": "TI", "m": "NTIpos", "f": "NTIpos", "c": "NTIneg"}
# 'a': antagonism = trophic interaction;
# 'f': facilitation = non-trophic positive interaction;
# 'c': competition = non-trophic negative interaction
num_species = 106
num_links = 4303  # num of links of every interaction type in the Chilean data
model = GLVmodel(num_species)

# Extract the fraction of each interaction type
fraction = {}
for interaction_code in list(interaction_type.keys()):
    file_name = f"chilean_{interaction_type[interaction_code]}"
    f = open(f"{data_path}/{file_name}.txt", "r")
    adj = []
    for line in f.readlines()[1:]:
        line = line.strip().split('\t')
        adj.append([int(elt) for elt in line[2:]])
    adj = np.array(adj)
    adj = np.reshape(adj, (len(adj), len(adj)))
    p = model.extract_proportion_of_interactions(adj, num_total_links=num_links, interaction_type=interaction_code)
    # print(f"p_{interaction_code} = {fraction}")
    fraction[interaction_code] = p

    # OUTPUT
    # 1358 links lead to p_a = 0.3156
    # 6 links lead to p_m = 0.0014
    # 160 links lead to p_f = 0.0372
    # 1456 links lead to p_c = 0.3384

print(fraction)
system(f"mkdir nested/chilean")
num_tests = 100
nl_range = np.arange(0., 1., 0.1)
# level = 0.7
p_stable = {}
for level in nl_range:
    system(f"mkdir nested/chilean/nl{level:.1f}")
    stability = []
    count = 0
    for run in tqdm(range(num_tests)):
        # Generate a matrix with random interaction strengths
        r, interaction_matrix, X_eq = model.generate_glv_params(fraction["a"], fraction["m"], fraction["f"],
                                                                fraction["c"], nestedness_level=level, nested_factor=1.0)
        # model.visualize_adjacency_matrix(interaction_matrix)

        stable = model.check_stability(interaction_matrix, X_eq)
        # if stable:
        #     count += 1
        #     print(count, stable)
        #     np.savetxt(f"nested/chilean/nl{level:.1f}/Run{run}.txt", interaction_matrix, fmt="%.8f")
        stability.append(stable)
    print(f"Nestedness level = {level:.1f} : {sum(stability)}/{num_tests} networks are stable !")
    p_stable[level] = sum(stability)/num_tests * 100
P_stable = np.c_[list(p_stable.keys()), list(p_stable.values())]
np.savetxt(f"nested/chilean/randomised_nested_model_{num_tests}samples.txt", P_stable, fmt="%.1f %.4f")

P_stable = np.loadtxt(f"nested/chilean/randomised_nested_model_{num_tests}samples.txt", dtype=float)
fig, ax = plt.subplots()
ax.tick_params(axis='both', labelsize=11)
ax.scatter(P_stable, marker="o", color="k")
plt.savefig(f"nested/chilean/randomised_nested_model_{num_tests}samples.svg", bbox_inches="tight")
