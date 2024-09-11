import setup_path
import gymnasium as gym

from typing import Union

import gym_drone
# from gym_drone.envs import *

import os
from utils import *
from git_utils import git_push
import argparse

from ray.rllib.algorithms.algorithm import Algorithm

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# os.environ["PYTHONWARNINGS"]="ignore::DeprecationWarning"

# Must start with origin
waypoint_variants = dict()

waypoint_variants["single"] = [
    np.array([0.0, 0.0, 5.0], dtype=np.float32),
    np.array([-10.0, -100.0, 5.0], dtype=np.float32)
]

waypoint_variants["multiple"] = [
    np.array([0.0, 0.0, 5.0], dtype=np.float32),
    np.array([15.0, 30.0, 5.0], dtype=np.float32),
    np.array([70.0, 35.0, 5.0], dtype=np.float32),
    np.array([75.0, 0.0, 5.0], dtype=np.float32),
    np.array([70.0, -35.0, 5.0], dtype=np.float32),
    np.array([15.0, -35.0, 5.0], dtype=np.float32),
]

waypoint_variants["obstacle"] = [
    np.array([0.0, 0.0, 5.0], dtype=np.float32),
    np.array([20.0, 0.0, 5.0], dtype=np.float32),
    np.array([20.0, 0.0, 18.0], dtype=np.float32),
    np.array([65.0, -5.0, 5.0], dtype=np.float32),
    np.array([75.0, 0.0, 5.0], dtype=np.float32)
]

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--custom", help="use the custom variant of the environment", action="store_true")
parser.add_argument("-a", "--algo", type=str, help="the algorithm to use for training", default="ppo",
                    choices=["ppo", "a2c", "a3c", "dqn", "td3", "ddpg", "sac", "marwil"])
parser.add_argument("-i", "--iter", type=int, help="the number of iterations to evaluate for", default=10)
parser.add_argument("-r", "--render", type=str, help="the render mode to use for evaluation", default="none")
parser.add_argument("-b", "--best", help="evaluate the best model over the iterations", action="store_true")
parser.add_argument("-w", "--waypoint_type", type=str, help="the waypoint type trained", default="random_single",
                    choices=["random_single", "random_multiple", "fixed_single", "fixed_multiple"])
# Change this
parser.add_argument("-v", "--waypoint_variant", type=str, help="the waypoint type to use for the environment",
                    choices=["single", "multiple", "obstacle"], default="single")

args = parser.parse_args()

env_var: str = "cust" if args.custom else "airsim"
total_iters: int = args.iter
algorithm: str = args.algo
render_mode: Union[str, None] = args.render
if render_mode == "none":
    render_mode = None
chkpt_root = "best_root" if args.best else "save_root"
waypoint_type = args.waypoint_type
waypoint_variant = args.waypoint_variant

end_at_start = True if waypoint_variant == "multiple" else False

env_config, env_ids = gym_drone.get_env_config(verbose=True, exp_waypts=waypoint_variants[waypoint_variant], end_at_start=end_at_start)

env_types = ["disc", "cont"]

shortest_routes = dict()

for env_type in env_types:
    print(f"Evaluating {env_type} environment...")
    env_id = f"drone-env-{env_type}-{env_var}"
    # env = gym.make(env_id, env_config=env_config)

    print("Using environment: {}".format(env_id))


    drone_env_config = {
        "waypoints": waypoint_variants[waypoint_variant][1:],
        "max_steps": None,
        "drone_name": "Drone1",
        "verbose": True,
        "momentum": False,
        "end_at_start": end_at_start
    }

    chkpt_path = f"training/{algorithm}/{env_type}/{env_var}/{waypoint_type}/{chkpt_root}"
    model = Algorithm.from_checkpoint(chkpt_path)
    eval_rewards, eval_lengths, success_rate, crashes, timeouts, route_list = evaluate_algorithm(model, env_id, epochs=total_iters, env_config=drone_env_config, render_mode=render_mode)

    print(f"Best reward over {total_iters} iterations: {max(eval_rewards)}")
    print(f"Average reward over {total_iters} iterations: {sum(eval_rewards) / total_iters}")
    print(f"Average episode length over {total_iters} iterations: {sum(eval_lengths) / total_iters}")
    print(f"Success rate over {total_iters} iterations: {success_rate}")
    print(f"Number of crashes: {crashes}/{total_iters}")
    print(f"Number of timeouts: {timeouts}/{total_iters}")

    # Check shortest route
    if len(route_list) == 0:
        print("No routes found")
        shortest_routes[env_type] = []

    route_lengths = []
    for route in route_list:
        route_lengths.append(sum([np.linalg.norm(route[i] - route[i+1]) for i in range(len(route) - 1)]))

    print(f"Shortest route length: {min(route_lengths)}")
    shortest_index = route_lengths.index(min(route_lengths))
    shortest_route = route_list[shortest_index]
    
    if waypoint_variant == "multiple":
        shortest_route.append(waypoint_variants[waypoint_variant][0])
    else:
        shortest_route.append(waypoint_variants[waypoint_variant][-1])

    shortest_routes[env_type] = shortest_route
    
    route_obj_filename = f"routes/{algorithm}/{waypoint_variant}/{env_type}_{env_var}_shortest_route.pkl"
    save_obj_file(route_obj_filename, shortest_routes[env_type])

route_plot_filename = f"routes/{algorithm}/{waypoint_variant}/{env_var}_shortest_route_top.png"
route_plot_filename_z = f"routes/{algorithm}/{waypoint_variant}/{env_var}_shortest_route_side.png"

if waypoint_variant == "obstacle":
    obstacles = [
        np.array([22.0, -22.0, 0.0, 63.0, 20.0, 15.0], dtype=np.float32),
        # np.array([50.0, 0.0, 5.0, 0.0, 0.0, 5.0], dtype=np.float32),
    ]
    plot_route_exp([waypoint_variants[waypoint_variant][0] ,waypoint_variants[waypoint_variant][-1]], shortest_routes, obstacles=obstacles, filename=route_plot_filename)
    plot_route_exp_z([waypoint_variants[waypoint_variant][0] ,waypoint_variants[waypoint_variant][-1]], shortest_routes, obstacles=obstacles, filename=route_plot_filename_z)
    
else:
    plot_route_exp(waypoint_variants[waypoint_variant], shortest_routes, filename=route_plot_filename)

git_push(f"Single UAV Exp: {waypoint_variant}, {algorithm.upper()}, {cap_first(env_type)}", True)