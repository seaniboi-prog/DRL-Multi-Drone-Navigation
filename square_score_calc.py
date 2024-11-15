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
    
# Example use case (commented out, as execution isn't needed here)
# Replace 'your_directory_path' with the actual directory path.
results_path = os.path.join("mtsp_results", "paths")
for file, full_path, details in list_files_with_paths(results_path):
    # print("File:", file)
    # print("Full Path:", full_path)
    # print("Details:", details)

    if os.path.exists(full_path):
        with open(full_path, 'rb') as f:
            mtsp_solver: AlgoMultiTSP = pickle.load(f)
        
        with open("squared_score.txt", "a") as output:
            output.write(f"Type: {details[0]}, Drones: {details[1][0]}, Algo: {details[2]}\n")
            output.write(f"Total Distance: {mtsp_solver.get_total_distance()}\n")
            output.write(f"Squared: {mtsp_solver.get_score_square()}\n")
            output.write(f"Root: {mtsp_solver.get_score_sqrt()}\n")
            output.write("\n\n")

    # print("")

# Function call is commented to prevent execution without a user-specified path.