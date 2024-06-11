from fileinput import filename
import sys
from ga_mtsp import GAMultiTSP
from aco_mtsp import ACOMultiTSP
from cvxpy_mtsp import CVXPYMultiTSP
from hill_mtsp import HillClimbMultiTSP
from tabu_mtsp import TabuSearchMultiTSP

from utils import *
from cities import get_cities
import string

# General Constants
city_type = "blocks"
# city_type = "random"

num_drones = 3

if city_type == "random":
    num_nodes = 20
    n_x = np.random.uniform(low=0, high=100, size=num_nodes-1)
    n_y = np.random.uniform(low=0, high=100, size=num_nodes-1)
    n_z = np.random.uniform(low=3, high=10, size=num_nodes-1)

    nodes = [np.array([n_x[i], n_y[i], n_z[i]]) for i in range(num_nodes-1)]
    nodes.insert(0, np.array([50, 50, 6])) # Start node
else:
    nodes = get_cities(city_type)
    num_nodes = len(nodes)

labels = [letter for letter in string.ascii_uppercase[:num_nodes]]

network = Network(num_drones, nodes, labels)

network.plot_network()

# HillClimb Constants
EPOCHS = 5000

hill_mtsp_solver = HillClimbMultiTSP(num_drones, nodes, labels)

hill_mtsp_solver.solve(EPOCHS)

plot_filename = compare_solution_scores(hill_mtsp_solver, city_type)

hill_mtsp_solver.plot_progress()
hill_mtsp_solver.plot_solution(filename=plot_filename)
hill_mtsp_solver.print_paths()
hill_paths_list = hill_mtsp_solver.get_paths_list(includeStart=False)
for i, path in enumerate(hill_paths_list):
    print(f"Drone {i+1}: ", end="")
    for j, node in enumerate(path):
        if j == len(path) - 1:
            print(f"({node[0]}, {node[1]}, {node[2]})", end="")
        else:
            print(f"({node[0]}, {node[1]}, {node[2]})", end=" -> ")
    print()
print()

# GA Constants
POPULATION_SIZE = 100
TOURNAMENT_SIZE = 10
GENERATIONS = 20
MUTATION_RATE = 0.05
CROSSOVER_RATE = 0.9
ELITISM = True

ga_mtsp_solver = GAMultiTSP(num_drones, nodes, labels)

ga_mtsp_solver.solve(GENERATIONS, MUTATION_RATE, TOURNAMENT_SIZE, POPULATION_SIZE, ELITISM, False)

plot_filename = compare_solution_scores(ga_mtsp_solver, city_type)

ga_mtsp_solver.plot_progress()
ga_mtsp_solver.plot_solution(filename=plot_filename)
ga_mtsp_solver.print_paths()
ga_paths_list = ga_mtsp_solver.get_paths_list(includeStart=False)
for i, path in enumerate(ga_paths_list):
    print(f"Drone {i+1}: ", end="")
    for j, node in enumerate(path):
        if j == len(path) - 1:
            print(f"({node[0]}, {node[1]}, {node[2]})", end="")
        else:
            print(f"({node[0]}, {node[1]}, {node[2]})", end=" -> ")
    print()
print()

# ACO Constants
RHO = 0.03
Q = 1
ALPHA = 1
BETA = 3
GEN_SIZE = None
LIMIT = 500
OPT2 = 30

aco_mtsp_solver = ACOMultiTSP(num_drones, nodes, labels)

aco_mtsp_solver.solve(ALPHA, BETA, RHO, Q, LIMIT, OPT2, False)

plot_filename = compare_solution_scores(aco_mtsp_solver, city_type)

aco_mtsp_solver.plot_progress()
aco_mtsp_solver.plot_solution(filename=plot_filename)
aco_mtsp_solver.print_paths()
aco_paths_list = aco_mtsp_solver.get_paths_list(includeStart=False)
for i, path in enumerate(aco_paths_list):
    print(f"Drone {i+1}: ", end="")
    for j, node in enumerate(path):
        if j == len(path) - 1:
            print(f"({node[0]}, {node[1]}, {node[2]})", end="")
        else:
            print(f"({node[0]}, {node[1]}, {node[2]})", end=" -> ")
    print()
print()

# Tabu Search Constants
MAX_TABU_SIZE = 10000
STOPPING_TURN = 500
NEIGHBOURHOOD_SIZE = 50

tabu_mtsp_solver = TabuSearchMultiTSP(num_drones, nodes, labels)

tabu_mtsp_solver.solve(NEIGHBOURHOOD_SIZE, MAX_TABU_SIZE, STOPPING_TURN)

plot_filename = compare_solution_scores(tabu_mtsp_solver, city_type)

tabu_mtsp_solver.plot_progress()
tabu_mtsp_solver.plot_solution(filename=plot_filename)
tabu_mtsp_solver.print_paths()
tabu_paths_list = tabu_mtsp_solver.get_paths_list(includeStart=False)
for i, path in enumerate(tabu_paths_list):
    print(f"Drone {i+1}: ", end="")
    for j, node in enumerate(path):
        if j == len(path) - 1:
            print(f"({node[0]}, {node[1]}, {node[2]})", end="")
        else:
            print(f"({node[0]}, {node[1]}, {node[2]})", end=" -> ")
    print()
print()

# CVXPY Constants
# cvxpy_mtsp_solver = CVXPYMultiTSP(num_drones, nodes, labels)

# cvxpy_mtsp_solver.solve(verbose=False)

# cvxpy_mtsp_solver.plot_solution()
# cvxpy_mtsp_solver.print_paths()

# Compare final distances
print("\nFinal Distances")
print(f"HillClimb: {round(hill_mtsp_solver.get_total_distance(), 2)}")
print(f"GA: {round(ga_mtsp_solver.get_total_distance(), 2)}")
print(f"ACO: {round(aco_mtsp_solver.get_total_distance(), 2)}")
print(f"Tabu Search: {round(tabu_mtsp_solver.get_total_distance(), 2)}")
# print(f"CVXPY: {round(cvxpy_mtsp_solver.get_total_distance(), 2)}")

print("\nFinal Scores")
print(f"HillClimb: {round(hill_mtsp_solver.get_score(), 2)}")
print(f"GA: {round(ga_mtsp_solver.get_score(), 2)}")
print(f"ACO: {round(aco_mtsp_solver.get_score(), 2)}")
print(f"Tabu Search: {round(tabu_mtsp_solver.get_score(), 2)}")
# print(f"CVXPY: {round(cvxpy_mtsp_solver.get_score(), 2)}")


fig, axs = plt.subplots(2, 2, figsize=(14, 10))

hill_mtsp_solver.plot_sub_solution(axs, (0,0), "HillClimb Solution")
ga_mtsp_solver.plot_sub_solution(axs, (0,1), "GA Solution")
aco_mtsp_solver.plot_sub_solution(axs, (1,0), "ACO Solution")
tabu_mtsp_solver.plot_sub_solution(axs, (1,1), "Tabu Search Solution")
# cvxpy_mtsp_solver.plot_sub_solution(axs, (1,1), "CVXPY Solution")

plt.suptitle("Comparing MultiTSP Solutions")

plt.tight_layout()

plt.savefig(f"plots/{city_type}/blocks_comparison.png")

plt.show()