from MultiTSP import *

import os
import re

def list_files_with_paths(base_path):
    """
    Traverse the given directory and list all files with their full paths.
    
    Parameters:
    base_path (str): The base directory to start traversal.
    
    Returns:
    list: A list of tuples where each tuple contains the file name and its full path.
    """    
    for root, _, files in os.walk(base_path):
        for file in files:
            full_path = os.path.join(root, file)
            details = full_path.split("\\")[2:4]
            algo = re.match(r'.*_(.*)_best_solution\.pkl', file)
            
            if algo:
                details.append(algo.group(1).upper())

            yield file, full_path, details


with open("squared_score.txt", "w") as output:
    output.write("Square Scores:\n")
    
results_path = os.path.join("mtsp_results", "paths")
for file, full_path, details in list_files_with_paths(results_path):

    if os.path.exists(full_path) and "random" in full_path:
        plot_path = full_path.replace("paths", "plots")
        plot_path = plot_path.replace("best_", "")
        plot_path = plot_path.replace("pkl", "png")
        print("Plot path:", plot_path)
        with open(full_path, 'rb') as f:
            mtsp_solver: AlgoMultiTSP = pickle.load(f)

        mtsp_solver.plot_solution(3, plot_path)
        
        # with open("squared_score.txt", "a") as output:
        #     output.write(f"Type: {details[0]}, Drones: {details[1][0]}, Algo: {details[2]}\n")
        #     output.write(f"Total Distance: {mtsp_solver.get_total_distance()}\n")
        #     output.write(f"Squared: {mtsp_solver.get_score_square()}\n")
        #     output.write(f"Root: {mtsp_solver.get_score_sqrt()}\n")
        #     output.write("\n\n")

