import sys
from ga_mtsp import GAMultiTSP
from aco_mtsp import ACOMultiTSP
from cvxpy_mtsp import CVXPYMultiTSP
from hill_mtsp import HillClimbMultiTSP

from utils import *
import string

# General Constants
num_drones = 3
num_nodes = 20
n_x = np.random.uniform(low=0, high=100, size=num_nodes-1)
n_y = np.random.uniform(low=0, high=100, size=num_nodes-1)
n_z = np.random.uniform(low=3, high=10, size=num_nodes-1)

nodes = [np.array([n_x[i], n_y[i], n_z[i]]) for i in range(num_nodes-1)]
nodes.insert(0, np.array([50, 50, 6])) # Start node
labels = [letter for letter in string.ascii_uppercase[:num_nodes]]

network = Network(num_drones, nodes, labels)

network.plot_network()

# HillClimb Constants
EPOCHS = 10000

hill_mtsp_solver = HillClimbMultiTSP(num_drones, nodes, labels)

hill_mtsp_solver.solve(EPOCHS)

hill_mtsp_solver.plot_progress()
hill_mtsp_solver.plot_solution()

# GA Constants
POPULATION_SIZE = 100
TOURNAMENT_SIZE = 10
GENERATIONS = 20
MUTATION_RATE = 0.05
CROSSOVER_RATE = 0.9
ELITISM = True

ga_mtsp_solver = GAMultiTSP(num_drones, nodes, labels)

ga_mtsp_solver.solve(GENERATIONS, MUTATION_RATE, TOURNAMENT_SIZE, POPULATION_SIZE, ELITISM, False)

ga_mtsp_solver.plot_progress()
ga_mtsp_solver.plot_solution()

# ACO Constants
RHO = 0.03
Q = 1
ALPHA = 1
BETA = 3
GEN_SIZE = None
LIMIT = 200
OPT2 = 30

aco_mtsp_solver = ACOMultiTSP(num_drones, nodes, labels)

aco_mtsp_solver.solve(ALPHA, BETA, RHO, Q, LIMIT, OPT2, False)

aco_mtsp_solver.plot_progress()
aco_mtsp_solver.plot_solution()

# CVXPY Constants
cvxpy_mtsp_solver = CVXPYMultiTSP(num_drones, nodes, labels)

cvxpy_mtsp_solver.solve(verbose=False)

cvxpy_mtsp_solver.plot_solution()

# Compare final distances
print("Final Distances")
print(f"HillClimb: {round(hill_mtsp_solver.get_total_distance(), 2)}")
print(f"GA: {round(ga_mtsp_solver.get_total_distance(), 2)}")
print(f"ACO: {round(aco_mtsp_solver.get_total_distance(), 2)}")
print(f"CVXPY: {round(cvxpy_mtsp_solver.get_total_distance(), 2)}")

print("\nFinal Scores")
print(f"HillClimb: {round(hill_mtsp_solver.get_score(), 2)}")
print(f"GA: {round(ga_mtsp_solver.get_score(), 2)}")
print(f"ACO: {round(aco_mtsp_solver.get_score(), 2)}")
print(f"CVXPY: {round(cvxpy_mtsp_solver.get_score(), 2)}")


fig, axs = plt.subplots(2, 2, figsize=(14, 10))

hill_mtsp_solver.plot_sub_solution(axs, (0,0), "HillClimb Solution")
ga_mtsp_solver.plot_sub_solution(axs, (0,1), "GA Solution")
aco_mtsp_solver.plot_sub_solution(axs, (1,0), "ACO Solution")
cvxpy_mtsp_solver.plot_sub_solution(axs, (1,1), "CVXPY Solution")

plt.suptitle("Comparing MultiTSP Solutions")

plt.tight_layout()

plt.show()