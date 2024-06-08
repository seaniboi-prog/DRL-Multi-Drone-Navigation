import gym_drone
import gymnasium as gym

import os
import time
import shutil
import random
import pprint
import torch
from tqdm import tqdm

from utils import *

import ray
from ray import air, tune, train
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.optuna import OptunaSearch

from ray.rllib.algorithms import ppo, dqn, sac, impala, marwil, bc
from ray.rllib.algorithms import a2c, a3c, td3, ddpg # deprecated

from push_notif import send_notification

import argparse

def get_tune_config(algo_name: str, tune_config: dict, env_name: str, env_config: Union[dict, None] = None):
    if algo_name == "ppo":
        config = ppo.PPOConfig()
        config = config.training(gamma=tune_config["gamma"],
                                 lr=tune_config["lr"],
                                 kl_coeff=tune_config["kl_coeff"],
                                 clip_param=tune_config["clip_param"],
                                 train_batch_size=tune_config["train_batch_size"]
                                )
    elif algo_name == "a2c":
        config = a2c.A2CConfig()
        config = config.training(gamma=tune_config["gamma"],
                                 lr=tune_config["lr"],
                                 train_batch_size=tune_config["train_batch_size"]
                                )
    elif algo_name == "a3c":
        config = a3c.A3CConfig()
        config = config.training(gamma=tune_config["gamma"],
                                 lr=tune_config["lr"],
                                 train_batch_size=tune_config["train_batch_size"]
                                )
    elif algo_name == "dqn":
        config = dqn.DQNConfig()
        config = config.training(gamma=tune_config["gamma"],
                                 lr=tune_config["lr"],
                                 train_batch_size=tune_config["train_batch_size"]
                                )
    elif algo_name == "ddpg":
        config = ddpg.DDPGConfig()
        config = config.training(gamma=tune_config["gamma"],
                                 lr=tune_config["lr"],
                                 train_batch_size=tune_config["train_batch_size"]
                                )
    elif algo_name == "td3":
        config = td3.TD3Config()
        config = config.training(gamma=tune_config["gamma"],
                                 lr=tune_config["lr"],
                                 train_batch_size=tune_config["train_batch_size"]
                                )
    elif algo_name == "sac":
        config = sac.SACConfig()
        config = config.training(gamma=tune_config["gamma"],
                                 lr=tune_config["lr"],
                                 train_batch_size=tune_config["train_batch_size"]
                                )
    elif algo_name == "impala":
        config = impala.ImpalaConfig()
        config = config.training(gamma=tune_config["gamma"],
                                 lr=tune_config["lr"],
                                 train_batch_size=tune_config["train_batch_size"]
                                )
    elif algo_name == "marwil":
        config = marwil.MARWILConfig()
        config = config.training(gamma=tune_config["gamma"],
                                 lr=tune_config["lr"],
                                 beta=tune_config["beta"],
                                 train_batch_size=tune_config["train_batch_size"]
                                )
    elif algo_name == "bc":
        config = bc.BCConfig()
        config = config.training(gamma=tune_config["gamma"],
                                 lr=tune_config["lr"],
                                 train_batch_size=tune_config["train_batch_size"]
                                )
    else:
        raise ValueError("Invalid algorithm name: {}".format(algo_name))
    
    if torch.cuda.is_available():
        num_gpus = NUM_PC_GPU
    else:
        num_gpus = NUM_LP_GPU
    
    config = config.resources(num_gpus=num_gpus, num_cpus_per_worker=1, num_learner_workers=1)
    config = config.rollouts(num_rollout_workers=0)
    config = config.framework("torch")

    if env_config is None:
        config = config.environment(env=env_name)
    else:
        config = config.environment(env=env_name, env_config=env_config)
    
    return config

def get_hyperparam_mutations(algo_name: str, batch_size: int) -> dict:
    if algo_name == "ppo":
        return {
            "gamma": tune.choice([0.7, 0.8, 0.9, 0.99]),
            "clip_param": tune.choice([0.1, 0.2, 0.3, 0.4, 0.5]),
            "lr": tune.choice([0.000001, 0.00001, 0.0001, 0.001]),
            "kl_coeff": tune.choice([0.1, 0.2, 0.3, 0.4, 0.5]),
            "train_batch_size": batch_size
        }
    elif algo_name == "a2c":
        return {
            "gamma": tune.choice([0.7, 0.8, 0.9, 0.99]),
            "lr": tune.choice([0.000001, 0.00001, 0.0001, 0.001]),
            "train_batch_size": batch_size
        }
    elif algo_name == "a3c":
        return {
            "gamma": tune.choice([0.7, 0.8, 0.9, 0.99]),
            "lr": tune.choice([0.000001, 0.00001, 0.0001, 0.001]),
            "train_batch_size": batch_size
        }
    elif algo_name == "dqn":
        return {
            "gamma": tune.choice([0.7, 0.8, 0.9, 0.99]),
            "lr": tune.choice([0.000001, 0.00001, 0.0001, 0.001]),
            "train_batch_size": batch_size
        }
    elif algo_name == "ddpg":
        return {
            "gamma": tune.choice([0.7, 0.8, 0.9, 0.99]),
            "lr": tune.choice([0.000001, 0.00001, 0.0001, 0.001]),
            "train_batch_size": batch_size
        }
    elif algo_name == "td3":
        return {
            "gamma": tune.choice([0.7, 0.8, 0.9, 0.99]),
            "lr": tune.choice([0.000001, 0.00001, 0.0001, 0.001]),
            "train_batch_size": batch_size
        }
    elif algo_name == "sac":
        return {
            "gamma": tune.choice([0.7, 0.8, 0.9, 0.99]),
            "lr": tune.choice([0.000001, 0.00001, 0.0001, 0.001]),
            "train_batch_size": batch_size
        }
    elif algo_name == "impala":
        return {
            "gamma": tune.choice([0.7, 0.8, 0.9, 0.99]),
            "lr": tune.choice([0.000001, 0.00001, 0.0001, 0.001]),
            "train_batch_size": batch_size
        }
    elif algo_name == "marwil":
        return {
            "beta": tune.choice([0.6, 0.7, 0.8, 0.9, 1.0]),
            "gamma": tune.choice([0.7, 0.8, 0.9, 0.99]),
            "lr": tune.choice([0.000001, 0.00001, 0.0001, 0.001]),
            "train_batch_size": batch_size
        }
    elif algo_name == "bc":
        return {
            "gamma": tune.choice([0.7, 0.8, 0.9, 0.99]),
            "lr": tune.choice([0.000001, 0.00001, 0.0001, 0.001]),
            "train_batch_size": batch_size
        }
    else:
        raise ValueError("Invalid algorithm name: {}".format(algo_name))

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--type", help="type of environment to train on", default="disc", choices=["cont", "disc"])
parser.add_argument("-c", "--custom", help="use the custom variant of the environment", action="store_true")
parser.add_argument("-a", "--algo", type=str, help="the algorithm to use for training", default="ppo",
                    choices=["ppo", "a2c", "a3c", "dqn", "td3", "ddpg", "sac", "impala", "marwil"])
parser.add_argument("-i", "--iter", type=int, help="the number of iterations to train for", default=20)
parser.add_argument("-m", "--momentum", help="use momentum in the environment", action="store_true")
parser.add_argument("-b", "--batch", type=int, help="the batch size to use for training", default=2048)
parser.add_argument("-p", "--parallel", type=int, help="the number of parallel trials to tune at a time", default=4, choices=[1, 2, 4, 8])
parser.add_argument("-r", "--restore", help="restore from a previous checkpoint", action="store_true")
parser.add_argument("-s", "--samples", type=int, help="total number of samples to take", default=20)
parser.add_argument("-u", "--update-best", help="update the best result config only with a higher reward average", action="store_true")
parser.add_argument("-w", "--waypoints", type=str, help="the waypoint type to use for the environment", default="random_single",
                    choices=["random_single", "random_multiple", "fixed_single", "fixed_multiple"])
parser.add_argument("--no-notif", help="disable sending notifications", action="store_true")
parser.add_argument("--max-steps", type=int, help="the maximum number of steps to run the environment")

args = parser.parse_args()

env_type: str = args.type
env_type_long: str = get_long_env_type(env_type)
env_var: str = "cust" if args.custom else "airsim"
algo_name: str = args.algo
iters: int = args.iter
momentum: bool = args.momentum
batch_size: int = args.batch
parallel_trials: int = args.parallel
restore: bool = args.restore
total_samples: int = args.samples
update_best_config: bool = args.update_best
waypoint_str: str = args.waypoints
waypoint_list = waypoint_str.split("_")
if waypoint_list[0] == "random":
    rand_waypoints = True
else:
    rand_waypoints = False
allow_notif: bool = not args.no_notif
max_steps = args.max_steps

env_config, env_ids = gym_drone.get_env_config(verbose=True, random_waypts=rand_waypoints, waypoint_type=waypoint_list[1], momentum=momentum, max_steps=max_steps)

# start Ray -- add `local_mode=True` here for debugging
print("Initialising ray...")
ray.shutdown()
ray.init()

# init directory in which to save checkpoints
tune_root = f"tune/{algo_name}/{env_type}/{env_var}/{waypoint_str}"
tune_result_root = os.path.join(tune_root, "tune_results")
cwd = os.getcwd()
tune_log_dir = os.path.join(cwd, tune_result_root)

# register the custom environment
print("Registering custom environment...")
select_env = f"drone-env-{env_type}-{env_var}"

if select_env not in env_ids:
    raise ValueError("Invalid environment name: {}".format(select_env))

print("Using environment: {}".format(select_env))

def train_env_algo(config):
        global algo_name, iters, select_env, env_config, waypoint_str, allow_notif, cwd
        
        config = get_tune_config(algo_name, config, select_env, env_config)

        algo = config.build()
            
        for _ in tqdm(range(iters), desc="Training ..."):
            result = algo.train()
            
        if allow_notif:
            send_notification("Training trail has Finished.", f"Training finished with average reward: {result['episode_reward_mean']}")

        train.report({
            "episode_reward_mean": result["episode_reward_mean"],
            "episode_reward_max": result["episode_reward_max"],
            "episode_reward_min": result["episode_reward_min"],
            "episode_len_mean": result["episode_len_mean"],
            "training_iteration": result["training_iteration"], 
            "timesteps_total": result["timesteps_total"],
            "episodes_total": result["episodes_total"],
        })

if torch.cuda.is_available():
    trainable_with_resources = tune.with_resources(train_env_algo, {"cpu": NUM_PC_CPU/parallel_trials, "gpu": NUM_PC_GPU/parallel_trials})
else:
    trainable_with_resources = tune.with_resources(train_env_algo, {"cpu": NUM_LP_CPU/parallel_trials, "gpu": NUM_LP_GPU/parallel_trials})

if restore:
    if not os.path.exists(tune_result_root):
        raise ValueError("Checkpoint directory does not exist.")
    
    print("Restoring from previous checkpoint...")
    if tune.Tuner.can_restore(tune_result_root):
        tuner = tune.Tuner.restore(tune_result_root, trainable_with_resources)
    else:
        raise ValueError("No checkpoint to restore from.")

else:
    shutil.rmtree(tune_result_root, ignore_errors=True)

    if not os.path.exists(os.path.dirname(tune_result_root)):
        os.makedirs(os.path.dirname(tune_result_root), exist_ok=True)

    stopping_criteria = {
        # "training_iteration": 2,
        # "timesteps_total": 10000,
        # "episodes_total": 1000
    }

    hyperparam_mutations = get_hyperparam_mutations(algo_name, batch_size)

    search_algo = ConcurrencyLimiter(OptunaSearch(), max_concurrent=4)

    tuner = tune.Tuner(
        trainable=trainable_with_resources,
        tune_config=tune.TuneConfig(
            metric="episode_reward_mean",
            mode="max",
            search_alg=search_algo,
            num_samples=total_samples
        ),
        param_space=hyperparam_mutations,
        run_config=air.RunConfig(
            name=f"tune_{algo_name}_{env_type}_{env_var}_{waypoint_str}",
            # stop=stopping_criteria,
            storage_path=tune_log_dir,
            verbose=2,
            # checkpoint_config=air.CheckpointConfig(
            #         checkpoint_frequency=5,
            #         checkpoint_at_end=True,
            # ),
        ),
    )

if allow_notif:
    send_notification("Tuning has Started.", f"Tuning {cap_first(env_type)} {env_var} {waypoint_str} with {algo_name}")

analysis = tuner.fit()

print("!!FINISHED!!")
ray.shutdown()

best_result = analysis.get_best_result()

if best_result is None or best_result.config is None or best_result.metrics is None:
    print("No best result found.")
    exit(1)

print("Best performing trial's final set of hyperparameters:\n")
pprint.pprint(
    {k: v for k, v in best_result.config.items() if k in hyperparam_mutations}
)

print("\nBest performing trial's final reported metrics:\n")

metrics_to_print = [
    "episode_reward_mean",
    "episode_reward_max",
    "episode_reward_min",
    "episode_len_mean",
]
pprint.pprint({k: v for k, v in best_result.metrics.items() if k in metrics_to_print})
best_avg_reward = best_result.metrics["episode_reward_mean"]

results_config = dict()
results_config["metrics"] = best_result.metrics
results_config["best_params"] = best_result.config

best_config_path = os.path.join(tune_root, "best_result_config.json")
if update_best_config:
    if os.path.exists(best_config_path):
        prev_best_config = load_dict_from_json(best_config_path)
        prev_best_reward = prev_best_config["metrics"]["episode_reward_mean"]

        if best_avg_reward < prev_best_reward:
            print("Best average reward did not improve.")
            exit(0)

save_dict_to_json(results_config, best_config_path)

if allow_notif:
    msg_1 = f"Tuning {cap_first(env_type)} {env_var} {waypoint_str} with {algo_name} has Finished"
    msg_2 = f"Best average reward: {best_result.metrics['episode_reward_mean']}"
    msg_3 = "Best hyperparameters:"
    msg_4 = ', '.join([f"{key}: {value}" for key, value in best_result.config.items()])
    send_notification(f"Tuning has Finished", f"{msg_1}\n{msg_2}\n{msg_3} {msg_4}")
