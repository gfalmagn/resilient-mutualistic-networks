from EcoResilience import GLVmodel

# Nested model parameters
N = 100  # Number of species
num_tests = 1
p_a, p_m, p_f, p_c = 0.1, 0.1, 0., 0.  # Interaction type proportions
nestedness_level = 0.2  # Proportion of core species
nested_factor = 10
model = GLVmodel(num_species=N)


stability = []
for _ in range(num_tests):
    # Generate parameters and assess stability
    r, interaction_matrix, X_eq = model.generate_glv_params(p_a, p_m, p_f, p_c, nestedness_level, nested_factor)
    # print(interaction_matrix)
    jacobian_matrix = model.compute_jacobian(interaction_matrix, X_eq, r)
    stable, eigenvalues = model.check_stability(jacobian_matrix)
    stability.append(stable)

    model.visualize_adjacency_matrix(interaction_matrix)
    # print("Equilibrium abundances:", X_eq)
    print(stable)
print(sum(stability))


# stability = []
# for _ in range(num_tests):
#     # Generate parameters and assess stability
#     r, interaction_matrix, X_eq = model.generate_lv_params(pA, pM, pF, pC, nestedness_level, nested_factor)
#     # visualize_network(interaction_matrix)
#
#     jacobian_matrix = model.compute_jacobian(interaction_matrix, X_eq, r)
#     stable, eigenvalues = model.check_stability(jacobian_matrix)
#     stability.append(stable)
#
#     model.visualize_adjacency_matrix(interaction_matrix)
#     # print("Equilibrium abundances:", X_eq)
#     print("Is the network stable?", stable)
# print(sum(stability))
