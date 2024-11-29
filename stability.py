from EcoResilience import GLVmodel
import numpy as np
from os import system

# Nested model parameters
N = 50  # Number of species
num_tests = 100
p_a, p_m, p_f, p_c = 0.4, 0.4, 0.1, 0.1  # Interaction type proportions
nestedness_level = 0.7  # Proportion of core species
nested_factor = 1.
model = GLVmodel(num_species=N)
system(f"mkdir nested/pa{p_a}_pm{p_m}_pf{p_f}_pc{p_c}")

save = True
stability = []
for run in range(num_tests):
    # Generate parameters and assess stability
    r, interaction_matrix, X_eq = model.generate_glv_params(p_a, p_m, p_f, p_c, nestedness_level, nested_factor)
    # print(interaction_matrix)
    stable = model.check_stability(interaction_matrix, X_eq)
    print(run, stable)
    if stable:
        np.savetxt(f"nested/pa{p_a}_pm{p_m}_pf{p_f}_pc{p_c}/N{N}_nl{nestedness_level}_nf{nested_factor:.1f}_Run{run}.txt",
                   interaction_matrix, fmt="%.8f")

    stability.append(stable)

    # model.visualize_adjacency_matrix(interaction_matrix)
    # print("Equilibrium abundances:", X_eq)
print(f"{sum(stability)}/{num_tests} networks (N={N}) are stable !")
