import os

from utils import *
import matplotlib.pyplot as plt
import numpy as np
from typing import Union

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def plot_episode_reward(rewards: Rewards, algo_name: str, env_name: str, env_type: str, reward_type: str, waypoint_type: str, display: bool, x_range: Union[list,None] = None, save_dir: str = 'plots'):
    
    save_file = f'{save_dir}/{reward_type}/reward_{reward_type}_plot_{env_type}_{env_name}_{algo_name}_{waypoint_type}.png'
    # os.makedirs(os.path.dirname(save_file), exist_ok=True)

    if reward_type == 'train':
        all_rewards, avg_rewards, avg_idxs = rewards.get_train_rewards()
    elif reward_type == 'eval':
        all_rewards, avg_rewards, avg_idxs = rewards.get_eval_rewards()
    else:
        raise ValueError(f'Invalid reward type: {reward_type}. Must be either "train" or "eval".')
    
    # Add first reward with average
    avg_rewards.insert(0, all_rewards[0])
    avg_idxs.insert(0, 1)
    
    # Plot rewards
    plt.figure(figsize=(12, 12))
    x = np.arange(1, len(all_rewards) + 1)
    plt.plot(x, all_rewards, color='blue', zorder=1)
    plt.plot(avg_idxs, avg_rewards, color='orange', linewidth=5, zorder=2)
    plt.xlabel('Episode')
    if x_range is not None:
        plt.xlim(x_range)
    plt.ylabel('Reward')
    waypoint_string = " ".join([cap_first(word) for word in waypoint_type.split("_")])
    plt.title(f'{cap_first(reward_type)} Rewards: {algo_name.upper()} in {cap_first(env_type)} {cap_first(env_name)} {waypoint_string}')
    plt.savefig(save_file)
    if display:
        plt.show()
    else:
        plt.close()
        
    return save_file
