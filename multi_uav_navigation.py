# Import Global Packages
import copy
import string
import airsim
import argparse
import multiprocessing

# Import Local Packages
from MultiPathPlanning.utils import *
from MultiPathPlanning.constants import *
from MultiPathPlanning.coordinates import get_waypoints
from MultiPathPlanning.navigate_drone import compute_single_episode

import MultiTSP as mtsp

import UAVNavigation.gym_drone.envs as airsim_envs

from ray.rllib.algorithms.algorithm import Algorithm

if __name__ == "__main__":
    # Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--action_type", help="type of action", default="disc", choices=["cont", "disc"])
    parser.add_argument("-c", "--custom", help="use custom version of environment", action="store_true")
    parser.add_argument("-r", "--rl_algo", type=str, help="the algorithm which the model was trained on", default="ppo",
                        choices=["ppo", "a2c", "a3c", "dqn", "td3", "ddpg", "sac", "impala", "marwil"])
    parser.add_argument("-m", "--mtsp_algo", type=str, help="the algorithm to use for solving the MTSP", default="ga",
                        choices=["ga", "aco", "cvxpy", "hill", "tabu"])
    parser.add_argument("-w", "--waypoint_type", type=str, help="which group of waypoints to choose")
    parser.add_argument("-p", "--waypoint_path", type=str, help="path to custom waypoints file", default="random_multiple")
    parser.add_argument("-b", "--best", help="use best model", action="store_true")
    parser.add_argument("-s", "--endAtStart", help="end at start", action="store_true")

    args = parser.parse_args()

    # Check for missing required arguments
    if args.waypoint_type is None:
        raise ValueError(F"{RED}Missing required argument: --waypoint_type (or -w){RESET}")
    
    # Get all arguments
    waypoint_type = args.waypoint_type
    mtsp_algo = args.mtsp_algo
    end_at_start = args.endAtStart
    
    rl_algo: str = args.rl_algo
    action_type: str = args.action_type
    waypoint_path: str = args.waypoint_path
    root_path: str = "best_root" if args.best else "save_root"
    env_variant: str = "cust" if args.custom else "airsim"
    
    is_custom = args.custom
    
    # Results Path
    results_root_path = os.path.join(os.getcwd(), "results", mtsp_algo, rl_algo, action_type, env_variant)

    # Retrieve inputs
    ## All waypoints
    waypoints = get_waypoints(waypoint_type)
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
    print(f"Number of drones: {no_drones}\n")
    
    redo = True
    
    mtsp_path_retries = []
    mtsp_path_scores = []

    while redo:
        # Generate paths
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
            mtsp_solver = mtsp.TabuSearchMultiTSP(no_drones, waypoints, labels)
            mtsp_solver.solve(NEIGHBOURHOOD_SIZE, MAX_TABU_SIZE, STOPPING_TURN)
        else:
            raise ValueError("Invalid MTSP algorithm. Must be one of: 'ga', 'aco', 'cvxpy', 'hill'")

        # Visualize paths
        mtsp_solver.plot_solution(pause=0, filename=os.path.join(results_root_path, f"{waypoint_type}_mtsp_solution.png"))

        print(f"Calculated Distance: {round(mtsp_solver.get_total_distance(), 2)}")
        print(f"Calculated Score: {round(mtsp_solver.get_score(), 2)}")
        
        # Retrieve paths
        mtsp_path_retries.append(copy.deepcopy(mtsp_solver.get_paths_list(includeStart=False)))
        mtsp_path_scores.append(mtsp_solver.get_score())
        
        if input("Do you want to redo the MTSP algorithm? (y/n): ").lower() != "y":
            redo = False
            
    # Retrieve best path
    best_path_idx = mtsp_path_scores.index(max(mtsp_path_scores))
    mtsp_paths = mtsp_path_retries[best_path_idx]
    print(f"Best Path Score: {mtsp_path_scores[best_path_idx]}")
    print("Executing the following paths:")
    for i, path in enumerate(mtsp_paths):
        print(f"Drone {i+1}: ", end="")
        for j, node in enumerate(path):
            if j == len(path) - 1:
                print(f"({node[0]}, {node[1]}, {node[2]})", end="")
            else:
                print(f"({node[0]}, {node[1]}, {node[2]})", end=" -> ")
    print()

    # Import DRL Model
    chkpt_path = os.path.join(os.getcwd(), f"UAVNavigation/training/{rl_algo}/{action_type}/{env_variant}/{waypoint_path}/{root_path}")
    uav_model = Algorithm.from_checkpoint(chkpt_path)

    # Retrieve environment class name
    if is_custom:
        drone_env_classname = f"DroneEnvCust_{action_type.capitalize()}"
    else:
        drone_env_classname = f"DroneEnv{action_type.capitalize()}"

    # Configuration of the Drone Agents
    uav_navigation_arg_sets = []

    for drone_id in range(no_drones):
        drone_env_config = {
            "waypoints": mtsp_paths[drone_id],
            "max_steps": None,
            "drone_name": vehicle_names[drone_id],
            "verbose": False,
            "end_at_start": end_at_start
        }
        drone_env_instace = getattr(airsim_envs, drone_env_classname)(env_config=drone_env_config)

        uav_navigation_arg_sets.append((drone_env_instace, uav_model))

    client.reset()

    # Execute drone navigation
    with multiprocessing.Pool(processes=len(uav_navigation_arg_sets)) as pool:
        results = pool.starmap(compute_single_episode, uav_navigation_arg_sets)

    print("All drones completed navigation")

    # Plot drone routes
    drone_routes = [result["route"] for result in results]
    total_distance = sum(result["total_distance"] for result in results)
    total_time = sum(result["total_time"] for result in results)
    mins, secs = divmod(total_time, 60)

    print(f"Total Distance Travelled: {total_distance}")
    print(f"Total Time Elapsed: {int(mins)} min/s, {secs:.2f} sec/s")
    
    plot_all_routes(waypoints, drone_routes, filename=os.path.join(results_root_path, f"{waypoint_type}_drone_routes.png"))

