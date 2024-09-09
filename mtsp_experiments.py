import sys

from MultiPathPlanning.utils import save_obj_file
from MultiTSP import *

import string
import argparse

from MultiPathPlanning.constants import *

if __name__ == "__main__":
    
    # Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--drones", type=int, help="number of drones", default=3)
    parser.add_argument("-r", "--random_nodes", type=int, help="number of random nodes")

    args = parser.parse_args()

    # General Constants
    num_drones = args.drones
    
    results_table_path = os.path.join(os.getcwd(), "mtsp_random_results.csv")
    
    # Check for missing required arguments

    num_nodes = args.random_nodes

    city_type = f"random_{num_nodes}"

    n_x = np.random.uniform(low=0, high=(num_nodes*20), size=num_nodes-1)
    n_y = np.random.uniform(low=0, high=(num_nodes*20), size=num_nodes-1)
    n_z = np.random.uniform(low=3, high=20, size=num_nodes-1)

    nodes = [np.array([n_x[i], n_y[i], n_z[i]]) for i in range(num_nodes-1)]
    nodes.insert(0, np.array([(num_nodes*10), (num_nodes*10), 6])) # Start node

    labels = get_labels(num_nodes)

    network = Network(num_drones, nodes, labels)

    network.plot_network()

    # HillClimb Constants
    EPOCHS = 5000

    hill_mtsp_solver = HillClimbMultiTSP(num_drones, nodes, labels)

    start_time = time.time()
    hill_mtsp_solver.solve(EPOCHS)
    end_time = time.time()

    elapsed_time = end_time - start_time
    
    solver_path = os.path.join("mtsp_results", "paths", city_type, f"{hill_mtsp_solver.n_drones}_drones", f"{city_type}_{hill_mtsp_solver.n_drones}_{hill_mtsp_solver.algorithm}_best_solution.pkl")
    save_obj_file(solver_path, hill_mtsp_solver)

    plot_filename = os.path.join("mtsp_results", "plots", city_type, f"{hill_mtsp_solver.n_drones}_drones", f"{city_type}_{hill_mtsp_solver.n_drones}_{hill_mtsp_solver.algorithm}_solution.png")
    results_table = update_mtsp_table(results_table_path, city_type, num_drones, "hill", hill_mtsp_solver, elapsed_time)

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

    ga_mtsp_solver = GAMultiTSP(num_drones, nodes, labels)

    start_time = time.time()
    ga_mtsp_solver.solve(GENERATIONS, MUTATION_RATE, TOURNAMENT_SIZE, POPULATION_SIZE, ELITISM, False)
    end_time = time.time()

    elapsed_time = end_time - start_time

    solver_path = os.path.join("mtsp_results", "paths", city_type, f"{ga_mtsp_solver.n_drones}_drones", f"{city_type}_{ga_mtsp_solver.n_drones}_{ga_mtsp_solver.algorithm}_best_solution.pkl")
    save_obj_file(solver_path, ga_mtsp_solver)

    plot_filename = os.path.join("mtsp_results", "plots", city_type, f"{ga_mtsp_solver.n_drones}_drones", f"{city_type}_{ga_mtsp_solver.n_drones}_{ga_mtsp_solver.algorithm}_solution.png")
    results_table = update_mtsp_table(results_table_path, city_type, num_drones, "ga", ga_mtsp_solver, elapsed_time)

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

    aco_mtsp_solver = ACOMultiTSP(num_drones, nodes, labels)

    start_time = time.time()
    aco_mtsp_solver.solve(ALPHA, BETA, RHO, Q, LIMIT, OPT2, False)
    end_time = time.time()

    elapsed_time = end_time - start_time

    solver_path = os.path.join("mtsp_results", "paths", city_type, f"{aco_mtsp_solver.n_drones}_drones", f"{city_type}_{aco_mtsp_solver.n_drones}_{aco_mtsp_solver.algorithm}_best_solution.pkl")
    save_obj_file(solver_path, aco_mtsp_solver)

    plot_filename = os.path.join("mtsp_results", "plots", city_type, f"{aco_mtsp_solver.n_drones}_drones", f"{city_type}_{aco_mtsp_solver.n_drones}_{aco_mtsp_solver.algorithm}_solution.png")
    results_table = update_mtsp_table(results_table_path, city_type, num_drones, "aco", aco_mtsp_solver, elapsed_time)

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

    tabu_mtsp_solver = TabuSearchMultiTSP(num_drones, nodes, labels)

    start_time = time.time()
    tabu_mtsp_solver.solve(NEIGHBOURHOOD_SIZE, MAX_TABU_SIZE, STOPPING_TURN)
    end_time = time.time()

    elapsed_time = end_time - start_time

    solver_path = os.path.join("mtsp_results", "paths", city_type, f"{tabu_mtsp_solver.n_drones}_drones", f"{city_type}_{tabu_mtsp_solver.n_drones}_{tabu_mtsp_solver.algorithm}_best_solution.pkl")
    save_obj_file(solver_path, tabu_mtsp_solver)

    plot_filename = os.path.join("mtsp_results", "plots", city_type, f"{tabu_mtsp_solver.n_drones}_drones", f"{city_type}_{tabu_mtsp_solver.n_drones}_{tabu_mtsp_solver.algorithm}_solution.png")
    results_table = update_mtsp_table(results_table_path, city_type, num_drones, "tabu", tabu_mtsp_solver, elapsed_time)


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
    
    # Compare final distances
    print("\nFinal Distances")
    print(f"HillClimb: {round(hill_mtsp_solver.get_total_distance(), 2)}")
    # print(f"GA: {round(ga_mtsp_solver.get_total_distance(), 2)}")
    print(f"ACO: {round(aco_mtsp_solver.get_total_distance(), 2)}")
    print(f"Tabu Search: {round(tabu_mtsp_solver.get_total_distance(), 2)}")

    print("\nFinal Scores")
    print(f"HillClimb: {round(hill_mtsp_solver.get_score(), 2)}")
    # print(f"GA: {round(ga_mtsp_solver.get_score(), 2)}")
    print(f"ACO: {round(aco_mtsp_solver.get_score(), 2)}")
    print(f"Tabu Search: {round(tabu_mtsp_solver.get_score(), 2)}")

    display_table(results_table, "MultiTSP Random Results")