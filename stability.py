from EcoResilience import GLVmodel

# Parameters
N = 100  # Number of species
num_tests = 1
pA, pM, pF, pC = 0.4, 0.4, 0., 0.  # Interaction type proportions
nestedness_level = 1.  # Proportion of core species
nested_factor = 1.5
model = GLVmodel(num_species=N)

stability = []
for _ in range(num_tests):
    # Generate parameters and assess stability
    r, interaction_matrix, X_eq = model.generate_lv_params(pA, pM, pF, pC, nestedness_level, nested_factor)
    # visualize_network(interaction_matrix)

    jacobian_matrix = model.compute_jacobian(interaction_matrix, X_eq, r)
    stable, eigenvalues = model.check_stability(jacobian_matrix)
    stability.append(stable)

    model.visualize_adjacency_matrix(interaction_matrix)
    # print("Equilibrium abundances:", X_eq)
    print("Is the network stable?", stable)
print(sum(stability))
