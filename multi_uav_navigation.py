# Import Global Packages
import copy
import string
import airsim
import pandas as pd
import argparse
import pyfiglet

# from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import Local Packages
from MultiPathPlanning.utils import *
from MultiPathPlanning.constants import *
from MultiPathPlanning.coordinates import get_waypoints, get_obstacles, get_specific_path
from MultiPathPlanning.register_envs import register_ray_gym_envs
from MultiPathPlanning.navigate_drone import compute_single_episode

from MultiTSP import *

import UAVNavigation.gym_drone.envs as airsim_envs

from ray.rllib.algorithms.algorithm import Algorithm

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

if __name__ == "__main__":
    # Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--action_type", help="type of action", default="disc", choices=["cont", "disc"])
    parser.add_argument("-c", "--custom", help="use custom version of environment", action="store_true")
    parser.add_argument("-r", "--rl_algo", type=str, help="the algorithm which the model was trained on", default="ppo",
                        choices=["ppo", "a2c", "a3c", "dqn", "td3", "ddpg", "sac", "impala", "marwil"])
    parser.add_argument("-m", "--mtsp_algo", type=str, help="the algorithm to use for solving the MTSP", default="aco",
                        choices=["ga", "aco", "cvxpy", "hill", "tabu"])
    parser.add_argument("-w", "--waypoint_type", type=str, help="which group of waypoints to choose")
    parser.add_argument("-p", "--waypoint_path", type=str, help="path to custom waypoints file", default="random_multiple")
    parser.add_argument("-b", "--best", help="use best model", action="store_true")
    parser.add_argument("-s", "--endAtStart", help="end at start", action="store_true")
    parser.add_argument("-f", "--fails", help="number of fails to allow", type=int, default=3)
    parser.add_argument("-sp", "--specific_path", help="specific path to use for waypoints", type=str)

    args = parser.parse_args()

    # Check for missing required arguments
    if args.waypoint_type is None:
        raise ValueError(F"{RED}Missing required argument: --waypoint_type (or -w){RESET}")
    
    # Get all arguments
    waypoint_type: str = args.waypoint_type
    mtsp_algo: str = args.mtsp_algo
    end_at_start = args.endAtStart
    specific_path: str = args.specific_path
    
    rl_algo: str = args.rl_algo
    action_type: str = args.action_type
    waypoint_path: str = args.waypoint_path
    root_path: str = "best_root" if args.best else "save_root"
    env_variant: str = "cust" if args.custom else "airsim"
    
    is_custom = args.custom

    # Retrieve inputs
    ## All waypoints
    waypoints = get_waypoints(waypoint_type)
    obstacles = get_obstacles(waypoint_type)
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

    # Results Path
    results_root_path = os.path.join(os.getcwd(), "results", mtsp_algo, f"{no_drones}_drones", rl_algo, action_type, env_variant)
    
    # Retrieve MTSP Algorithm
    mtsp_solver: AlgoMultiTSP = load_obj_file(os.path.join("mtsp_results", "paths", waypoint_type, f"{no_drones}_drones", f"{waypoint_type}_{no_drones}_{mtsp_algo}_best_solution.pkl"))
    
    # Visualize paths
    mtsp_solver.plot_solution(pause=5)

    print(f"Calculated Distance: {round(mtsp_solver.get_total_distance(), 2)}")
    print(f"Calculated Score: {round(mtsp_solver.get_score(), 2)}")
    
    # Retrieve paths
    if specific_path is not None:
        mtsp_paths = mtsp_solver.get_paths_list(includeStart=False)
        print(f"Path Score: {mtsp_solver.get_score()}")
    else:
        mtsp_paths = get_specific_path(specific_path)
    print("Executing the following paths:")
    for i, path in enumerate(mtsp_paths):
        print(f"Drone {i+1}: ", end="")
        for j, node in enumerate(path):
            if j == len(path) - 1:
                print(f"({node[0]}, {node[1]}, {node[2]})", end="")
            else:
                print(f"({node[0]}, {node[1]}, {node[2]})", end=" -> ")
    print()
    
    register_ray_gym_envs(waypoint_type=waypoint_type, end_at_start=end_at_start)

    # Import DRL Model
    chkpt_path = os.path.join(os.getcwd(), f"UAVNavigation/training/{rl_algo}/{action_type}/{env_variant}/{waypoint_path}/{root_path}")
    uav_model = Algorithm.from_checkpoint(chkpt_path)

    # Retrieve environment class name
    if is_custom:
        drone_env_classname = f"DroneEnvCust_{action_type.capitalize()}"
    else:
        drone_env_classname = f"DroneEnv{action_type.capitalize()}"


    # Execute drone navigation
    # with multiprocessing.Pool(processes=len(uav_navigation_arg_sets)) as pool:
    #     results = pool.starmap(compute_single_episode, uav_navigation_arg_sets)
    
    no_fails = 0
    max_fails = args.fails
    
    while True:
        # Configuration of the Drone Agents
        uav_navigation_arg_sets = []

        for drone_id in range(no_drones):
            drone_idx = (drone_id + no_fails) % no_drones
            drone_env_config = {
                "waypoints": mtsp_paths[drone_id],
                "max_steps": None,
                "drone_name": vehicle_names[drone_id],
                "verbose": True,
                "momentum": False,
                "end_at_start": end_at_start
            }
            drone_env_instace = getattr(airsim_envs, drone_env_classname)(env_config=drone_env_config)

            uav_navigation_arg_sets.append((drone_env_instace, uav_model))
            
        client.reset()
        start_time = time.time()
        
        results = []
        with ThreadPoolExecutor(max_workers=len(uav_navigation_arg_sets)) as executor:
            futures = [executor.submit(compute_single_episode, env, model) for env, model in uav_navigation_arg_sets]
            for future in as_completed(futures):
                results.append(future.result())
                
        elapsed_time = time.time() - start_time
        
        solved = all(result["status"] == "solved" for result in results)
        
        if solved or no_fails >= max_fails:
            break
        else:
            no_fails += 1
            print(f"Failed attempt: {no_fails}")

    print("All drones completed navigation")

    # Plot drone routes
    drone_routes = [result["route"] for result in results]
    total_distance = sum(result["total_distance"] for result in results)
    average_time = sum(result["total_time"] for result in results) / len(results)
    mins, secs = divmod(elapsed_time, 60)
    
    solved = all(result["status"] == "solved" for result in results)

    print(f"Total Distance Travelled: {total_distance}")
    print(f"Total Time Elapsed: {int(mins)} min/s, {secs:.2f} sec/s")
    
    plot_all_routes(waypoints, drone_routes, obstacles, filename=os.path.join(results_root_path, f"{waypoint_type}_{no_drones}_drone_routes.png"))

    # Save results
    results_path = os.path.join(results_root_path, f"{waypoint_type}_{no_drones}_results.pkl")
    save_obj_file(results_path, results)

    if solved:
        # Add results to Table
        results_table_path = os.path.join(os.getcwd(), "multi_uav_nav_results.csv")
        results_table = update_multiuav_table(results_table_path, waypoint_type, no_drones, mtsp_algo, rl_algo, action_type, env_variant, results, mtsp_solver)

        display_table(results_table, "Multi UAV Navigation Results")
    
    git_push(f"Exp: {waypoint_type}, {no_drones} drones, {mtsp_algo.upper()}, {rl_algo.upper()}, {action_type}", True)
    
    ascii_art = pyfiglet.figlet_format("COMPLETE", font='slant')
    print(ascii_art)