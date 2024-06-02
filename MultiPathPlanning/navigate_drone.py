from typing import Union
from gymnasium import Env
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.policy import Policy

def compute_single_episode(env: Env, model: Union[Algorithm, Policy], initial_obs):
    obs = initial_obs
    done = False
    success: bool = False
    total_reward: float = 0.0
    episode_length: int = 0
    while not done:
        action = model.compute_single_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_length += 1
        
        if terminated or truncated:
            done = True
            success = info["solved"]
        
        total_reward += float(reward)
    return total_reward, episode_length, success