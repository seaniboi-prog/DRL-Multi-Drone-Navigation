from MultiPathPlanning.utils import *
from MultiPathPlanning.constants import *
from MultiPathPlanning.coordinates import get_waypoints, get_obstacles

import os

waypoint_type = "blocks"
mtsp_algo = "aco"
rl_algo = "ppo"
action_type = "disc"
env_type = "airsim"

results_root_path = os.path.join(os.getcwd(), "results", mtsp_algo, rl_algo, action_type, env_type)

waypoints = get_waypoints(waypoint_type)
obstacles = get_obstacles(waypoint_type)

results = load_obj_file(os.path.join(results_root_path, f"{waypoint_type}_results.pkl"))

drone_routes = [result["route"] for result in results]

plot_all_routes(waypoints, drone_routes, obstacles, os.path.join(results_root_path, f"{waypoint_type}_drone_routes.png"))