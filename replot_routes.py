from MultiPathPlanning.utils import *
from MultiPathPlanning.constants import *
from MultiPathPlanning.coordinates import get_waypoints, get_obstacles

import os
import pandas as pd

waypoint_type = "blocks"
no_drones = 3
mtsp_algo = "aco"
rl_algo = "ppo"
action_type = "disc"
env_variant = "airsim"

results_root_path = os.path.join(os.getcwd(), "results", mtsp_algo, rl_algo, action_type, env_variant)

waypoints = get_waypoints(waypoint_type)
obstacles = get_obstacles(waypoint_type)

results = load_obj_file(os.path.join(results_root_path, f"{waypoint_type}_results.pkl"))

drone_routes = [result["route"] for result in results]

plot_all_routes(waypoints, drone_routes, obstacles, os.path.join(results_root_path, f"{waypoint_type}_drone_routes.png"))

total_distance = sum(result["total_distance"] for result in results)
elapsed_time = max(result["total_time"] for result in results)
average_time = sum(result["total_time"] for result in results) / len(results)

results_table_path = "results_table.csv"

# Check if the file exists and read the CSV, otherwise create an empty DataFrame with specified columns
if os.path.exists(results_table_path):
    results_table = pd.read_csv(results_table_path)
else:
    results_table = pd.DataFrame(columns=["Slug", "Waypoint Type", "No Drones", "MTSP Algorithm", "RL Algorithm", "Action Type", "Total Distance", "Total Time"])

# Construct the slug and row dictionary
slug = f"{waypoint_type}_{mtsp_algo}_{rl_algo}_{action_type}_{env_variant}_{no_drones}"
row = {
    "Slug": slug,
    "Waypoint Type": waypoint_type.capitalize(),
    "No Drones": no_drones,
    "MTSP Algorithm": mtsp_algo.upper(),
    "RL Algorithm": rl_algo.upper(),
    "Action Type": action_type.capitalize(),
    "Total Distance": total_distance,
    "Total Time": elapsed_time,
    "Average Time": average_time
}

# Check if the slug exists in the 'Slug' column and update or append the row
if slug in results_table["Slug"].values:
    results_table.loc[results_table["Slug"] == slug, :] = pd.DataFrame([row]).values
else:
    results_table = pd.concat([results_table, pd.DataFrame([row])], ignore_index=True)

# Save the updated DataFrame back to the CSV file
results_table.to_csv(results_table_path, index=False)

display_table(results_table, "Multi UAV Navigation Results")