from MultiPathPlanning.utils import *
from MultiPathPlanning.constants import *
from MultiPathPlanning.coordinates import get_waypoints, get_obstacles

from MultiTSP import *

import os

waypoint_type = "mountains"
no_drones = 3
mtsp_algo = "tabu"
rl_algo = "ppo"
action_type = "cont"
env_variant = "airsim"

results_root_path = os.path.join(os.getcwd(), "results", mtsp_algo, f"{no_drones}_drones", rl_algo, action_type, env_variant)

waypoints = get_waypoints(waypoint_type)
obstacles = get_obstacles(waypoint_type)

results = load_obj_file(os.path.join(results_root_path, f"{waypoint_type}_{no_drones}_results.pkl"))

drone_routes = [result["route"] for result in results]

plot_all_routes(waypoints, drone_routes, obstacles, os.path.join(results_root_path, f"{waypoint_type}_{no_drones}_drone_routes.png"))

results_table_path = os.path.join(os.getcwd(), "multi_uav_nav_results.csv")
mtsp_solver: AlgoMultiTSP = load_obj_file(os.path.join("mtsp_results", "paths", waypoint_type, f"{no_drones}_drones", f"{waypoint_type}_{no_drones}_{mtsp_algo}_best_solution.pkl"))
results_table = update_multiuav_table(results_table_path, waypoint_type, no_drones, mtsp_algo, rl_algo, action_type, env_variant, results, mtsp_solver)

display_table(results_table, "Multi UAV Navigation Results")