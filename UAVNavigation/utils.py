from typing import Union
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.policy import Policy
from ray import tune
import gymnasium as gym
from gymnasium import Env
import gym_drone
import random
import numpy as np
import torch
import json
import os
import sys
import math

from rich.console import Console
from rich.table import Table

from tqdm import tqdm

from ray.rllib.algorithms import ppo, dqn, sac, impala, marwil, bc
# from ray.rllib.algorithms import a2c, a3c, td3, ddpg # deprecated
# from rllib_a2c.a2c import a2c
# from rllib_a3c.a3c import a3c
# from rllib_ddpg.ddpg import ddpg
# from rllib_td3.td3 import td3

from pickle_utils import save_obj_file, load_obj_file

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

NUM_PC_CPU = 4
NUM_PC_GPU = 1
NUM_LP_CPU = 8
NUM_LP_GPU = 0

def compute_single_episode(env: Env, model: Union[Algorithm, Policy]):
    obs, info = env.reset()
    done = False
    total_reward: float = 0.0
    episode_length: int = 0
    while not done:
        action = model.compute_single_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_length += 1
        
        if terminated or truncated:
            done = True
            status = info["status"]
            route = info["route"]
        
        total_reward += float(reward)
    return total_reward, episode_length, status, route

def evaluate_algorithm(model: Union[Algorithm,Policy], env_id: str, epochs: int = 10, env_config: Union[dict,None] = None, render_mode: Union[str, None] = None):
    if env_config is None:
        env = gym.make(env_id)
    else:
        env_config["render_mode"] = render_mode
        env = gym.make(env_id, env_config=env_config)
    
    rewards_list: list[float] = []
    episode_lengths: list[int] = []
    route_list = []
    successes = 0
    timeouts = 0
    crashes = 0
    for _ in tqdm(range(epochs), desc="Evaluating..."):
        cum_reward, episode_len, status, route = compute_single_episode(env, model)
        rewards_list.append(cum_reward)
        episode_lengths.append(episode_len)
        route_list.append(route)
        if status == "solved":
            successes += 1
        elif status == "timed_out":
            timeouts += 1
        elif status == "crashed":
            crashes += 1
            
    env.close()
    
    success_rate = successes / epochs
    
    return rewards_list, episode_lengths, success_rate, crashes, timeouts, route_list

def get_algo_config(algo_name: str, env_name:str, env_config: Union[dict, None] = None, batch_size: int = 1024, params: dict = {}):
    if algo_name == "ppo":
        config = ppo.PPOConfig()
        config = config.training(kl_coeff=params.get("kl_coeff", 0.3), clip_param=params.get("clip_param", 0.3))
    # elif algo_name == "a2c":
    #     config = a2c.A2CConfig()
    # elif algo_name == "a3c":
    #     config = a3c.A3CConfig()
    elif algo_name == "dqn":
        config = dqn.DQNConfig()
    # elif algo_name == "ddpg":
    #     config = ddpg.DDPGConfig()
    # elif algo_name == "td3":
    #     config = td3.TD3Config()
    elif algo_name == "sac":
        config = sac.SACConfig()
    elif algo_name == "impala":
        config = impala.ImpalaConfig()
    elif algo_name == "marwil":
        config = marwil.MARWILConfig()
        config = config.training(beta=params.get("beta", 1.0))
    elif algo_name == "bc":
        config = bc.BCConfig()
    else:
        raise ValueError("Invalid algorithm name: {}".format(algo_name))
    
    config = config.training(gamma=params.get("gamma", 0.7), lr=params.get("lr", 0.0001), train_batch_size=batch_size)
    
    if torch.cuda.is_available():
        config = config.resources(num_gpus=1, num_gpus_per_worker=1)
        config = config.rollouts(num_rollout_workers=0)
        config = config.framework("torch")
    else:
        config = config.rollouts(num_rollout_workers=0)
        config = config.framework("torch")

    if env_config is None:
        config = config.environment(env=env_name)
    else:
        config = config.environment(env=env_name, env_config=env_config)
    
    return config

def cap_first(string: str):
    return string[0].upper() + string[1:]

def get_long_env_type(env_type: str) -> str:
    if env_type == "cont":
        return "continuous"
    elif env_type == "disc":
        return "discrete"
    elif env_type == "cust":
        return "custom"
    else:
        raise ValueError("Invalid environment type: {}".format(env_type))
    
def get_elapsed_time(seconds: float):
    mins, secs = divmod(seconds, 60)
    hrs, mins = divmod(mins, 60)
    
    return hrs, mins, secs

def average_list(lst: list):
    return sum(lst) / len(lst)

def save_dict_to_json(dictionary, filename: str):
    with open(filename, "w") as f:
        json.dump(dictionary, f, indent=4)
        
def load_dict_from_json(filename: str) -> dict:
    with open(filename, "r") as f:
        return json.load(f)
        
def get_last_checkpoint(path):
    # Get a list of all subdirectories in the specified path
    subdirectories = [subdir for subdir in os.listdir(path) if os.path.isdir(os.path.join(path, subdir))]
    
    # Sort the list of subdirectories by their names (ascending order)
    subdirectories.sort()

    # Get the last subdirectory (latest alphabetically)
    if subdirectories:
        return subdirectories[-1]
    else:
        return None
    
def get_cust_counterpart(env_name: str):
    if env_name == "drone-env-disc-v2":
        cust_env = "env1"

    elif env_name == "drone-env-cont-v1":
        cust_env = "env2"
        
    return cust_env

class PrintCounter:
    """
    Class to track and print status with counter in the terminal, overwriting previous line.
    """
    def __init__(self):
        self.prev_status = None
        self.counter = 0
        self.break_line = True

    def print_status(self, status):
        """
        Prints the status with a counter, overwriting the previous line if necessary.

        Args:
        status: The status string to print.
        """
        if status == self.prev_status or self.prev_status is None:
            # Same status, update counter and overwrite previous line
            self.counter += 1
            print(f"\r{status}: {self.counter}", end="")
            sys.stdout.flush()  # Force flush to update terminal immediately
            self.break_line = True
        else:
            # Reset counter and print new status on a new line
            if self.break_line:
                print()
                self.break_line = False
            self.counter = 1
            print(f"\r{status}: {self.counter}", end="")
            self.prev_status = status
    
    def reset(self):
        """
        Resets the counter and previous status.
        """
        self.counter = 0
        self.prev_status = None

class Rewards:
    def __init__(self):
        # Main reward lists
        self.train_rewards = []
        self.eval_rewards = []

        # Average rewards lists
        self.train_avg_rewards = []
        self.train_avg_idxs = []
        self.eval_avg_rewards = []
        self.eval_avg_idxs = []

    def extend_train_reward(self, rewards: list, avg_reward: float, avg_index: int) -> None:
        self.train_rewards.extend(rewards)
        self.train_avg_rewards.append(avg_reward)
        self.train_avg_idxs.append(avg_index)

    def extend_eval_reward(self, rewards: list, avg_reward: float, avg_index: int) -> None:
        self.eval_rewards.extend(rewards)
        self.eval_avg_rewards.append(avg_reward)
        self.eval_avg_idxs.append(avg_index)

    def get_train_rewards(self):
        return self.train_rewards, self.train_avg_rewards, self.train_avg_idxs
    
    def get_max_train_reward(self):
        return max(self.train_rewards)
    
    def get_min_train_reward(self):
        return min(self.train_rewards)
    
    def get_avg_train_reward(self):
        return average_list(self.train_rewards)
    
    def get_eval_rewards(self):
        return self.eval_rewards, self.eval_avg_rewards, self.eval_avg_idxs
    
    def save_rewards(self, path: str) -> None:
        save_obj_file(path, self)

    def restore_rewards(self, path: str) -> None:
        loaded_obj = load_obj_file(path)
        self.__dict__.update(loaded_obj.__dict__)

def display_table(df, title):
    table = Table(title=title)
    rows = df.values.tolist()
    rows = [[str(el) for el in row] for row in rows]
    columns = df.columns.tolist()

    for column in columns:
        table.add_column(column)

    for row in rows:
        table.add_row(*row, style='bright_green')

    console = Console()
    console.print(table)

def pop_first_element(arr):
    return arr[0], arr[1:]

def plot_route(targets, drone_path, obstacles=[], filename=None):
    
    plt.figure(figsize=(8, 8))
    
    # Plot the obstacles
    for obs in obstacles:
        x_min, y_min, _, x_max, y_max, _ = obs
        plt.gca().add_patch(Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, fill=True, color='grey', alpha=0.8, zorder=1))

    for i, path in enumerate(drone_path):
        route = np.array(path)
        plt.plot(route[:, 0], route[:, 1], c="green", zorder=2, label=f"Drone {i+1}")

    np_targets = np.array(targets)
    start, np_targets = pop_first_element(np_targets)
    plt.scatter(start[0], start[1], c='red', marker='x', s=60, zorder=3, label="Start")
    plt.scatter(np_targets[:, 0], np_targets[:, 1], c='black', s=60, marker='x', zorder=3, label="Waypoints")

    plt.legend()
    plt.title("Drone Routes")
    plt.xlabel("X")
    plt.ylabel("Y")
    if filename is not None:
        if os.path.dirname(filename) != '' and not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename)
    
    plt.show(block=False)  # Set block=False to allow code execution to continue
    plt.pause(5)

    # Close the plot window
    plt.close()