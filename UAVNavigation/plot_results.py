import os

from utils import *
import matplotlib.pyplot as plt
import numpy as np
from typing import Union

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Union

# TODO: Font size
def plot_episode_reward(rewards: Rewards, algo_name: str, env_name: str, env_type: str, reward_type: str, waypoint_type: str, display: bool, x_range: Union[list, None] = None, save_dir: str = 'plots'):
    save_file = f'{save_dir}/{reward_type}/reward_{reward_type}_plot_{env_type}_{env_name}_{algo_name}_{waypoint_type}.png'
    os.makedirs(os.path.dirname(save_file), exist_ok=True)

    if reward_type == 'train':
        all_rewards, avg_rewards, avg_idxs = rewards.get_train_rewards()
    elif reward_type == 'eval':
        all_rewards, avg_rewards, avg_idxs = rewards.get_eval_rewards()
    else:
        raise ValueError(f'Invalid reward type: {reward_type}. Must be either "train" or "eval".')

    if x_range is not None:
        new_avg_rewards = []
        new_avg_idxs = []
        for i in range(len(avg_rewards)):
            if x_range[0] <= avg_idxs[i] <= x_range[1]:
                new_avg_rewards.append(avg_rewards[i])
                new_avg_idxs.append(avg_idxs[i])

        avg_rewards = new_avg_rewards
        avg_idxs = new_avg_idxs
        all_rewards = all_rewards[x_range[0]:avg_idxs[-1]]

    # Add first reward with average
    avg_rewards.insert(0, all_rewards[0])
    avg_idxs.insert(0, 1)

    # Plot rewards
    plt.figure(figsize=(12, 12))
    x = np.arange(1, len(all_rewards) + 1)
    plt.plot(x, all_rewards, color='blue', zorder=1)
    plt.plot(avg_idxs, avg_rewards, color='orange', linewidth=5, zorder=2)
    
    # Set labels with increased fontsize
    plt.xlabel('Episode', fontsize=24)
    plt.ylabel('Reward', fontsize=24)
    
    # Set the title with increased fontsize
    waypoint_string = " ".join([cap_first(word) for word in waypoint_type.split("_")])
    plt.title(f'{cap_first(reward_type)} Rewards: {algo_name.upper()} in {cap_first(env_type)} {cap_first(env_name)} {waypoint_string}', fontsize=28)
    
    # Adjust tick parameters for both axes
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.tick_params(axis='both', which='minor', labelsize=16)
    
    # Save and display plot
    plt.savefig(save_file)
    if display:
        plt.show()
    else:
        plt.close()

    return save_file


if __name__ == '__main__':
    waypoint_str = "random_single"
    reward_type = "train"
    env_var = "airsim"
    
    algorithm = "ppo"
    env_type = "disc"
    x_range = [0, 2000]
    
    reward_root = f"rewards/{algorithm}/{env_type}/{env_var}/{waypoint_str}/rewards.pkl"
    all_episode_rewards = Rewards()
    all_episode_rewards.restore_rewards(reward_root)
    
    plot_episode_reward(all_episode_rewards, algorithm, env_var, env_type, reward_type, waypoint_str, display=True, x_range=x_range)
    
    algorithm = "ppo"
    env_type = "cont"
    x_range = [0, 2000]
    
    reward_root = f"rewards/{algorithm}/{env_type}/{env_var}/{waypoint_str}/rewards.pkl"
    all_episode_rewards = Rewards()
    all_episode_rewards.restore_rewards(reward_root)
    
    plot_episode_reward(all_episode_rewards, algorithm, env_var, env_type, reward_type, waypoint_str, display=True, x_range=x_range)
    
    algorithm = "marwil"
    env_type = "disc"
    x_range = None
    
    reward_root = f"rewards/{algorithm}/{env_type}/{env_var}/{waypoint_str}/rewards.pkl"
    all_episode_rewards = Rewards()
    all_episode_rewards.restore_rewards(reward_root)
    
    plot_episode_reward(all_episode_rewards, algorithm, env_var, env_type, reward_type, waypoint_str, display=True, x_range=x_range)
    
    algorithm = "marwil"
    env_type = "cont"
    x_range = None
    
    reward_root = f"rewards/{algorithm}/{env_type}/{env_var}/{waypoint_str}/rewards.pkl"
    all_episode_rewards = Rewards()
    all_episode_rewards.restore_rewards(reward_root)
    
    plot_episode_reward(all_episode_rewards, algorithm, env_var, env_type, reward_type, waypoint_str, display=True, x_range=x_range)
    
    algorithm = "sac"
    env_type = "disc"
    x_range = None
    
    reward_root = f"rewards/{algorithm}/{env_type}/{env_var}/{waypoint_str}/rewards.pkl"
    all_episode_rewards = Rewards()
    all_episode_rewards.restore_rewards(reward_root)
    
    plot_episode_reward(all_episode_rewards, algorithm, env_var, env_type, reward_type, waypoint_str, display=True, x_range=x_range)
    
    algorithm = "sac"
    env_type = "cont"
    x_range = None
    
    reward_root = f"rewards/{algorithm}/{env_type}/{env_var}/{waypoint_str}/rewards.pkl"
    all_episode_rewards = Rewards()
    all_episode_rewards.restore_rewards(reward_root)
    
    plot_episode_reward(all_episode_rewards, algorithm, env_var, env_type, reward_type, waypoint_str, display=True, x_range=x_range)