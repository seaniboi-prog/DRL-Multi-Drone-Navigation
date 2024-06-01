import setup_path
import gymnasium as gym

from typing import Union

import gym_drone
# from gym_drone.envs import *

import os
from utils import *
import argparse

from ray.rllib.algorithms.algorithm import Algorithm

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--type", help="type of environment to train on", default="disc", choices=["cont", "disc", "cust"])
parser.add_argument("-e", "--env", type=str, help="the environment version to train on (e.g. v1)", default="v2")
parser.add_argument("-a", "--algo", type=str, help="the algorithm to use for training", default="ppo",
                    choices=["ppo", "a2c", "a3c", "dqn", "td3", "ddpg"])
parser.add_argument("-i", "--iter", type=int, help="the number of iterations to evaluate for", default=10)
parser.add_argument("-r", "--render", type=str, help="the render mode to use for evaluation", default="none")
parser.add_argument("-b", "--best", help="evaluate the best model over the iterations", action="store_true")
parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
parser.add_argument("-w", "--waypoints", type=str, help="the waypoint type to use for the environment", default="random_single",
                    choices=["random_single", "random_multiple", "fixed_single", "fixed_multiple"])

args = parser.parse_args()

env_no: str = f"en{args.env}"
env_type: str = args.type
env_type_long: str = get_long_env_type(env_type)
total_iters: int = args.iter
algorithm: str = args.algo
render_mode: Union[str, None] = args.render
if render_mode == "none":
    render_mode = None
chkpt_root = "best_root" if args.best else "save_root"
verbose: bool = args.verbose
waypoint_str: str = args.waypoints
waypoint_list = waypoint_str.split("_")
if waypoint_list[0] == "random":
    rand_waypoints = True
else:
    rand_waypoints = False

env_config, env_ids = gym_drone.get_env_config(verbose=verbose, random_waypts=rand_waypoints, waypoint_type=waypoint_list[1])

env_id = f"drone-env-{env_type}-{args.env}"
# env = gym.make(env_id, env_config=env_config)

if env_id not in env_ids:
    raise ValueError("Invalid environment name: {}".format(env_id))

print("Using environment: {}".format(env_id))

chkpt_path = f"training/{algorithm}/{env_type}/{env_no}/{waypoint_str}/{chkpt_root}"
model = Algorithm.from_checkpoint(chkpt_path)
eval_rewards, eval_lengths, success_rate = evaluate_algorithm(model, env_id, epochs=total_iters, env_config=env_config, render_mode=render_mode)

# model_root = f"models/{algorithm}/{env_type}/{env_no}/{waypoint_str}/model/model.pt"
# model = torch.load(model_root)
# eval_rewards, eval_lengths, success_rate = evaluate_algorithm_pytorch(model_root, env_id, epochs=total_iters, env_config=env_config, render_mode=render_mode)

print(f"Best reward over {total_iters} iterations: {max(eval_rewards)}")
print(f"Average reward over {total_iters} iterations: {sum(eval_rewards) / total_iters}")
print(f"Average episode length over {total_iters} iterations: {sum(eval_lengths) / total_iters}")
print(f"Success rate over {total_iters} iterations: {success_rate}")
