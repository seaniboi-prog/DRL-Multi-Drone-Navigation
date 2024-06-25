import pandas as pd
from utils import load_dict_from_json, display_table

import os
import fnmatch

def find_files(directory, filename):
    matches = []
    for root, dirnames, filenames in os.walk(directory):
        dirnames[:] = [d for d in dirnames if d != 'tune_results']
        for file in fnmatch.filter(filenames, filename):
            matches.append(os.path.join(root, file))
    return matches

directory_to_search = 'tune'
filename_to_find = 'best_result_config.json'

found_files = find_files(directory_to_search, filename_to_find)

# Crete a DataFrame to store the results
results_table_path = os.path.join(os.getcwd(), "tune_results_table.csv")

if os.path.exists(results_table_path):
    results_table = pd.read_csv(results_table_path)
else:
    results_table = pd.DataFrame(columns=["Slug", "Algorithm", "Action", "Waypoints", "Gamma", "LR", "Reward Mean", "Reward Max", "Training Time"])

substrings_to_remove = ["tune", "random_", "cust", "best_result_config.json"]

for file_path in found_files:
    # Extract the parameters from the file path
    slug = file_path
    for substring in substrings_to_remove:
        slug = slug.replace(substring, "")
    slug = slug.replace("\\", "/")
    slug = slug.replace("/", "_")
    slug = slug.replace("__", "_")
    slug = slug.strip("_")

    # Load the dictionary from the JSON file
    result_dict = load_dict_from_json(file_path)

    algo, action, waypoint_type = slug.split("_")[:3]

    # Construct the row dictionary
    row = {
        "Slug": slug,
        "Algorithm": algo,
        "Action": action,
        "Waypoints": waypoint_type,
        "Gamma": result_dict["best_params"]["gamma"],
        "LR": result_dict["best_params"]["lr"],
        "Reward Mean": result_dict["metrics"]["episode_reward_mean"],
        "Reward Max": result_dict["metrics"]["episode_reward_max"],
        "Training Time": result_dict["metrics"]["time_total_s"]
    }

    # Check if the slug exists in the 'Slug' column and update or append the row
    if slug in results_table["Slug"].values:
        results_table.loc[results_table["Slug"] == slug, :] = pd.DataFrame([row]).values
    else:
        results_table = pd.concat([results_table, pd.DataFrame([row])], ignore_index=True)

# Save the updated DataFrame back to the CSV file
results_table.to_csv(results_table_path, index=False)

display_table(results_table, "Tune Results Table")