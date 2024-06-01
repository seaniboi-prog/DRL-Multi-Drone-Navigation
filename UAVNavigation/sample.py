import setup_path
import gymnasium as gym

from gymnasium import Env

import gym_drone
import pprint

def run_one_episode (env: Env, max_steps, verbose=False):
    env.reset()
    sum_reward = 0.0

    for i in range(max_steps):
        action = env.action_space.sample()

        if verbose:
            print("action:", action)

        state, reward, terminated, truncated, info = env.step(action)
        sum_reward += float(reward)

        if verbose:
            env.render()

        if terminated or truncated:
            if verbose:
                print("done @ step {}".format(i))

            break

    if verbose:
        print("cumulative reward", sum_reward)

    return sum_reward

env_config, env_ids = gym_drone.get_env_config()

# first, create the custom environment and run it for one episode
max_steps = env_config["max_steps"]
env = gym.make("drone-env-disc-v4", env_config=env_config)
sum_reward = run_one_episode(env, max_steps, verbose=False)

# next, calculate a baseline of rewards based on random actions
# (no policy)
history = []

for _ in range(10):
    sum_reward = run_one_episode(env, max_steps, verbose=False)
    history.append(sum_reward)

avg_sum_reward = sum(history) / len(history)
print("\nbaseline cumulative reward: {:6.2}".format(avg_sum_reward))