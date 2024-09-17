import gym_drone
import gymnasium as gym

import os
import sys
import time
import copy
import shutil
import pprint

from pickle_utils import save_obj_file, load_obj_file
# from push_notif import send_notification, send_notif_image
from git_utils import git_push
from plot_results import plot_episode_reward
from utils import *

from telegram_utils import wait_for_response, send_message, send_photo
CONTINUE_MESSAGE = "Would you like to continue training? (Y/N)"
CONTINUE_TIMEOUT = 10 # minutes
CONTINUE_INTERVAL = 20 # seconds

import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.utils.framework import try_import_torch
from ray.tune.logger import pretty_print

import argparse

import traceback
from msgpackrpc.error import RPCError

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--type", help="type of environment to train on", default="disc", choices=["cont", "disc"])
parser.add_argument("-c", "--custom", help="use the custom variant of the environment", action="store_true")
parser.add_argument("-a", "--algo", type=str, help="the algorithm to use for training", default="ppo",
                    choices=["ppo", "a2c", "a3c", "dqn", "td3", "ddpg", "sac", "impala", "marwil"])
parser.add_argument("-i", "--iter", type=int, help="the number of iterations to train for", default=10)
parser.add_argument("-b", "--batch", type=int, help="the batch size to use for training", default=2048)
parser.add_argument("-r", "--render", type=str, help="the render mode to use for evaluation", default="none")
parser.add_argument("-m", "--momentum", help="use momentum in the environment", action="store_true")
parser.add_argument("-w", "--waypoints", type=str, help="the waypoint type to use for the environment", default="random_single",
                    choices=["random_single", "random_multiple", "fixed_single", "fixed_multiple"])
parser.add_argument("--tuned", help="use tuned hyperparameters", action="store_true")
parser.add_argument("--no-notif", help="disable sending notifications", action="store_true")
parser.add_argument("--max-steps", type=int, help="the maximum number of steps to run the environment")

args = parser.parse_args()

print("Importing torch...")
torch, nn = try_import_torch()

env_type: str = args.type
env_type_long: str = get_long_env_type(env_type)
env_var: str = "cust" if args.custom else "airsim" 
algo_name: str = args.algo
batch_size: int = args.batch
tuned_params: bool = args.tuned
allow_notif: bool = not args.no_notif
render_mode: Union[str, None] = args.render
if render_mode == "none":
    render_mode = None
momentum: bool = args.momentum
waypoint_str: str = args.waypoints
waypoint_list = waypoint_str.split("_")
if waypoint_list[0] == "random":
    rand_waypoints = True
else:
    rand_waypoints = False
waypoint_cap = " ".join([cap_first(word) for word in waypoint_list])
max_steps = args.max_steps

env_config, env_ids = gym_drone.get_env_config(verbose=True, random_waypts=rand_waypoints, waypoint_type=waypoint_list[1], momentum=momentum, max_steps=max_steps)

print("Training on {} environment {} with {} waypoints using {} algorithm for {} iterations".format(env_type_long, cap_first(env_var), waypoint_cap, algo_name.upper(), args.iter))

# init directory in which to save checkpoints
chkpt_root = f"training/{algo_name}/{env_type}/{env_var}/{waypoint_str}/save_root"
best_root = f"training/{algo_name}/{env_type}/{env_var}/{waypoint_str}/best_root"
best_avg_pkl = f"training/{algo_name}/{env_type}/{env_var}/{waypoint_str}/best_avg_reward.pkl"
model_root = f"models/{algo_name}/{env_type}/{env_var}/{waypoint_str}"
reward_root = f"rewards/{algo_name}/{env_type}/{env_var}/{waypoint_str}/rewards.pkl"
shutil.rmtree(chkpt_root, ignore_errors=True, onerror=None)
shutil.rmtree(model_root, ignore_errors=True, onerror=None)
shutil.rmtree(reward_root, ignore_errors=True, onerror=None)

if not os.path.exists(os.path.dirname(chkpt_root)):
    os.makedirs(os.path.dirname(chkpt_root), exist_ok=True)
if not os.path.exists(os.path.dirname(model_root)):
    os.makedirs(os.path.dirname(model_root), exist_ok=True)
if not os.path.exists(os.path.dirname(reward_root)):
    os.makedirs(os.path.dirname(reward_root), exist_ok=True)

# start Ray -- add `local_mode=True` here for debugging
print("Initialising ray...")
ray.init(local_mode=False)

# register the custom environment
print("Registering custom environment...")
select_env = f"drone-env-{env_type}-{env_var}"

if select_env not in env_ids:
    raise ValueError("Invalid environment name: {}".format(select_env))

print("Using environment: {}".format(select_env))

if tuned_params:
    print("Using tuned hyperparameters...")
    tune_root = f"tune/{algo_name}/{env_type}/cust/{waypoint_str}/best_result_config.json"
    try:
        tune_config = load_dict_from_json(tune_root)
        tune_params = tune_config["best_params"]
        pprint.pprint(tune_params)
    except:
        print("Error loading tuned hyperparameters from: {}".format(tune_root))
        tune_params = {}
else:
    tune_params = {}

config = get_algo_config(algo_name, select_env, env_config, batch_size, tune_params)
print("\nAlgorithm Config:")
# print(pretty_print(config.to_dict()))

print("Building algorithm config...")
algo: Algorithm = config.build()
    
training_durations = []
all_episode_rewards = Rewards()
train_episode_count = 0
eval_episode_count = 0
print("\nStarting training...")
if allow_notif:
    # send_notification("Training Started", "Started Training using {} in {} {} {} waypoints"
    #                 .format(algo_name.upper(), cap_first(env_type), env_var, waypoint_cap))
    send_message("Started Training using {} in {} {}".format(algo_name.upper(), cap_first(env_type), env_var))

total_iters: int = args.iter
i = 0
try:
    best_avg_reward = load_obj_file(best_avg_pkl)
except:
    best_avg_reward = -float("inf")
while i < total_iters:
    try:
        # Train the algorithm
        print("Training Iteration {}:".format((i+1)))
        start_time = time.time()
        result = algo.train()
        end_time = time.time() - start_time
        training_durations.append(end_time)
        hrs, mins, secs = get_elapsed_time(end_time)
        
        # Save Checkpoint
        try:
            chkpt_file = algo.save_checkpoint(chkpt_root)
        except:
            print("Creating new checkpoint...")
            chkpt_file = algo.save(chkpt_root)
            chkpt_file = algo.save_checkpoint(chkpt_root)
        
        # Export Model and Checkpoint
        model_file = algo.export_model(["model", "checkpoint"], model_root)

        print(pretty_print(result))
        
        # Retrieve Rewards and Episode Lengths
        episode_rewards = result["hist_stats"]["episode_reward"]
        episode_lengths = result["hist_stats"]["episode_lengths"]
        
        train_episode_count += len(episode_rewards)
        
        all_episode_rewards.extend_train_reward(episode_rewards, result["episode_reward_mean"], train_episode_count)

        # Display the results
        length_info = {
            "max_length": max(episode_lengths),
            "min_length": min(episode_lengths),
            "avg_length": average_list(episode_lengths),
        }
        
        reward_info = {
            "max_reward": result["episode_reward_max"],
            "min_reward": result["episode_reward_min"],
            "avg_reward": result["episode_reward_mean"],
        }
        
        print("Training Finished")
        print("Time Taken: {} hr/s {} min/s {} sec/s".format(hrs, mins, secs))
        print("Max Reward: {}".format(reward_info["max_reward"]))
        print("Avg Reward: {}".format(reward_info["avg_reward"]))
        print("Min Reward: {}".format(reward_info["min_reward"]))
        
        if reward_info["avg_reward"] > best_avg_reward:
            best_avg_reward = copy.deepcopy(reward_info["avg_reward"])
            
            try:
                best_file = algo.save_checkpoint(best_root)
            except:
                print("Creating new best checkpoint...")
                best_file = algo.save(best_root)
                best_file = algo.save_checkpoint(best_root)
                
            save_obj_file(best_avg_pkl, best_avg_reward)
            
            print("New Best Reward: {:.2f}".format(reward_info["avg_reward"]))
        
        # Plot Training Rewards
        train_plot = plot_episode_reward(all_episode_rewards, algo_name, env_var, env_type, "train", waypoint_str, display=False)
        
        # Send Training Notification
        if allow_notif:
            notif_title = "Training Iteration {} Finished".format(result["training_iteration"])
            notif_msg1 = "Finished Training using {} in {} {} with {} waypoints at {} hr/s {} min/s {:.2f} sec/s".format(algo_name.upper(), cap_first(env_type), cap_first(env_var), waypoint_cap, hrs, mins, secs)
            notif_msg2 = "Average reward of {:.2f} with a range between {:.2f} and {:.2f}".format(reward_info["avg_reward"], reward_info["min_reward"], reward_info["max_reward"])
            notif_msg3 = "Average episode length of {:.2f} with a range between {} and {}".format(length_info["avg_length"], length_info["min_length"], length_info["max_length"])

            # send_notif_image(notif_title, f"{notif_msg1}\n{notif_msg2}\n{notif_msg3}", train_plot)
            # send_notification(notif_title, f"{notif_msg1}\n{notif_msg2}\n{notif_msg3}")
            send_message(notif_title)
            send_message(f"{notif_msg1}\n{notif_msg2}\n{notif_msg3}")
        
        # Evaluate the algorithm
        print("\nEvaluating Algorithm...")
        eval_rewards, eval_lengths, success_rate, crashes, timeouts, route_list, time_list = evaluate_algorithm(algo, select_env, epochs=5, env_config=env_config, render_mode=render_mode)
        
        eval_episode_count += len(eval_rewards)
        
        all_episode_rewards.extend_eval_reward(eval_rewards, average_list(eval_rewards), eval_episode_count)
        
        # Plot Evaluation Rewards
        eval_plot = plot_episode_reward(all_episode_rewards, algo_name, env_var, env_type, "eval", waypoint_str, display=False)
        
        # Send Evaluation Notification
        if allow_notif:
            notif_title = "Evaluation Iteration {} Finished with {} episodes".format(result["training_iteration"], len(eval_rewards))
            notif_msg1 = "Average reward of {:.2f} with a range between {:.2f} and {:.2f}".format(average_list(eval_rewards), min(eval_rewards), max(eval_rewards))
            notif_msg2 = "Average episode length of {:.2f} with a range between {} and {}".format(average_list(eval_lengths), min(eval_lengths), max(eval_lengths))
            notif_msg3 = "Success Rate: {:.2%}, Crashes: {}, Timeouts: {}".format(success_rate, crashes, timeouts)
            
            # send_notif_image(notif_title, f"{notif_msg1}\n{notif_msg2}\n{notif_msg3}", eval_plot)
            # send_notification(notif_title, f"{notif_msg1}\n{notif_msg2}\n{notif_msg3}")

        # Save Rewards
        all_episode_rewards.save_rewards(reward_root)

        if allow_notif:
            # Push the results to the git repository
            git_push("Training Iteration {} Finished using {} in {} {} with {} waypoints"
                    .format(result["training_iteration"], algo_name.upper(), cap_first(env_type), cap_first(env_var), waypoint_cap),
                    pull=True)

        # Print Result
        # print(pretty_print(result))
        print()

        i += 1

        if i >= total_iters and allow_notif:
            confirm = wait_for_response(CONTINUE_MESSAGE, CONTINUE_TIMEOUT, CONTINUE_INTERVAL)
            if confirm:
                print("Received confirmation, continuing training...")
                # send_notification("Continuing Training", "Received confirmation, continuing training...")
                total_iters += 1

    except KeyboardInterrupt:
        print(f"Training Iteration {(i+1)} Interrupted")
        break

    except RPCError as err:
        print(f"Training Iteration {(i+1)} Failed with RPCError")
        print(f"Error Type: {type(err)}")
        print(f"Error Message: {err}")
        print("Skipping to next iteration...")
        if allow_notif:
            # send_notification(f"Training Iterantion {(i+1)} Failed",
            #                 f"Failed with MsgpackRPC Error: {err}\nSkipping to next iteration...")
            send_message(f"Training Iterantion {(i+1)} Failed with MsgpackRPC Error\nSkipping to next iteration...")
        traceback.print_exc()
        sys.exit(1)
    
    except Exception as err:
        print(f"Training Iteration {(i+1)} Failed with General Error")
        print(f"Error Type: {type(err)}")
        print(f"Error Message: {err}")
        print("Skipping to next iteration...")
        if allow_notif:
            # send_notification(f"Training Iterantion {(i+1)} Failed",
            #                 f"Failed with General Error: {err}\nSkipping to next iteration...")
            send_message(f"Training Iterantion {(i+1)} Failed with General Error\nSkipping to next iteration...")
        traceback.print_exc()
        sys.exit(1)

algo.stop()

print("Shutting down ray...")
ray.shutdown()

print("Finished training on {} environment {} using {} algorithm".format(env_type_long, env_var, algo_name.upper()))

hrs, mins, secs = get_elapsed_time(average_list(training_durations))
print("Average Training Duration: {} hr/s {} min/s {:.2f} sec/s".format(hrs, mins, secs))
hrs, mins, secs = get_elapsed_time(sum(training_durations))
print("Total Time Taken: {} hr/s {} min/s {:.2f} sec/s".format(hrs, mins, secs))

print("Total Max Reward: {}".format(all_episode_rewards.get_max_train_reward()))
print("Total Avg Reward: {}".format(all_episode_rewards.get_avg_train_reward()))
print("Total Min Reward: {}".format(all_episode_rewards.get_min_train_reward()))