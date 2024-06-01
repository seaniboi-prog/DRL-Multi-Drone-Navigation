# Import Global Packages
import numpy as np
import matplotlib.pyplot as plt
import argparse
import string
import airsim

# Import Local Packages
# from MultiPathPlanning.cust_classes import DronePaths
from MultiPathPlanning.utils import *
from MultiPathPlanning.waypoints import get_waypoints
from MultiPathPlanning.constants import *

import MultiTSP as mtsp

import UAVNavigation.gym_drone.envs as airsim_envs

from ray.rllib.algorithms.algorithm import Algorithm

# Parse Arguments
parser = argparse.ArgumentParser()
parser.add_argument("-t", "--action_type", help="type of action", default="disc", choices=["cont", "disc"])
parser.add_argument("-r", "--rl_algo", type=str, help="the algorithm which the model was trained on", default="ppo",
                    choices=["ppo", "a2c", "a3c", "dqn", "td3", "ddpg", "sac", "impala", "marwil"])
parser.add_argument("-m", "--mtsp_algo", type=str, help="the algorithm to use for solving the MTSP", default="ga",
                    choices=["ga", "aco", "cvxpy", "hill", "tabu"])
parser.add_argument("-w", "--waypoint_type", type=str, help="which group of waypoints to choose")

args = parser.parse_args()

# Check for missing required arguments
if args.waypoint_type is None:
    raise ValueError("Missing required argument: --waypoint_type (or -w)")

# Retrieve inputs
## All waypoints
waypoints = get_waypoints(args.waypoint_type)
num_nodes = len(waypoints)
labels = [letter for letter in string.ascii_uppercase[:num_nodes]]

## No. of drones
client = airsim.MultirotorClient()
try:
    client.confirmConnection()
except Exception as e:
    raise ConnectionError("Failed to connect to AirSim simulator.") from e

vehicle_names = client.listVehicles()
no_drones = len(vehicle_names)

# Generate paths
mtsp_algo = args.mtsp_algo
if mtsp_algo == "ga":
    mtsp_solver = mtsp.GAMultiTSP(no_drones, waypoints, labels)
    mtsp_solver.solve(GENERATIONS, MUTATION_RATE, TOURNAMENT_SIZE, POPULATION_SIZE, ELITISM, cont=False)
elif mtsp_algo == "aco":
    mtsp_solver = mtsp.ACOMultiTSP(no_drones, waypoints, labels)
    mtsp_solver.solve(ALPHA, BETA, RHO, Q, LIMIT, OPT2, cont=False)
elif mtsp_algo == "cvxpy":
    mtsp_solver = mtsp.CVXPYMultiTSP(no_drones, waypoints, labels)
    mtsp_solver.solve(verbose=False)
elif mtsp_algo == "hill":
    mtsp_solver = mtsp.HillClimbMultiTSP(no_drones, waypoints, labels)
    mtsp_solver.solve(EPOCHS)
elif mtsp_algo == "tabu":
    # TODO: Implement Tabu Search
    raise NotImplementedError("Tabu Search not yet implemented")
else:
    raise ValueError("Invalid MTSP algorithm. Must be one of: 'ga', 'aco', 'cvxpy', 'hill'")

# Visualize paths
mtsp_solver.plot_progress()
mtsp_solver.plot_solution()

print(f"Calculated Distance: {round(mtsp_solver.get_total_distance(), 2)}")
print(f"Calculated Score: {round(mtsp_solver.get_score(), 2)}")

# Retrieve paths
mtsp_paths = mtsp_solver.get_paths_list()

# drone_paths = DronePaths()
# for drone_id, path in enumerate(mtsp_paths):
#     drone_paths.add_path(path)
# drone_paths.save_paths("drone_paths.pkl")

# Configure drones
# TODO: Implement drone configuration
chkpt_path = os.path.join(os.getcwd(), f"UAVNavigation/training/{args.rl_algo}/{args.action_type}/random_multiple/save_root")
uav_model = Algorithm.from_checkpoint(chkpt_path)

for drone_id in range(len(mtsp_paths)):
    drone_env_config = {
        "waypoints": mtsp_paths[drone_id],
        "max_steps": None,
        "drone_name": vehicle_names[drone_id],
    }
    

# Execute drone navigation


# Results

