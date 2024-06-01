import setup_path
import gymnasium as gym

from gymnasium import Env

import gym_drone

def run_one_episode (env: Env, max_steps, verbose=False):
    env.reset()
    sum_reward = 0.0

    for i in range(max_steps):
        action = int(input("Enter action: (0-4)"))

        if verbose:
            print("action:", action)

        state, reward, terminated, truncated, info = env.step(action)
        sum_reward += float(reward)

        if verbose:
            print("reward:", reward)

        if terminated or truncated:
            env.reset()
            if verbose:
                print("done @ step {}".format(i))

            break

    if verbose:
        print("cumulative reward", sum_reward)

    return sum_reward

# first, create the custom environment and run it for one episode

env_config, env_ids = gym_drone.get_env_config()

env = gym.make("drone-env-disc-v2", env_config=env_config)
sum_reward = run_one_episode(env, env_config["max_steps"], verbose=True)

# next, calculate a baseline of rewards based on random actions
# (no policy)
# history = []

# for _ in range(10):
#     sum_reward = run_one_episode(env, max_steps, verbose=False)
#     history.append(sum_reward)

# avg_sum_reward = sum(history) / len(history)
# print("\nbaseline cumulative reward: {:6.2}".format(avg_sum_reward))