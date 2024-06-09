from typing import Union
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.policy import Policy
from UAVNavigation.gym_drone.envs import DroneEnv_Base
from MultiPathPlanning.constants import RED, GREEN, RESET

def compute_single_episode(env: DroneEnv_Base, model: Union[Algorithm, Policy]) -> dict:
    # Initialize variables
    done = False
    total_reward: float = 0.0
    episode_length: int = 0

    # Take off and get initial observation
    obs, info = env.reset(options={"_reset": False})
    
    print("Starting episode of ", env.drone_name)

    # Start episode
    while not done:
        action = model.compute_single_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_length += 1
        
        if terminated or truncated:
            done = True
        
        total_reward += float(reward)

    results = {
        "total_reward": total_reward,
        "episode_length": episode_length,
        "status": info["status"],
        "route": info["route"],
        "total_distance": info["distance_travelled"],
        "total_time": info["time_elapsed"],
    }

    print(f"\n{env.drone_name} RESULTS:")
    if info["status"] == "solved":
        print(f"{GREEN}{env.drone_name} COMPLETED SUCCESSFULLY!{RESET}")
    elif info["status"] == "timed_out":
        print(f"{RED}{env.drone_name} TIMED OUT{RESET}")
    elif info["status"] == "crashed":
        print(f"{RED}{env.drone_name} CRASHED{RESET}")
    print(f"Total Reward: {results['total_reward']}")
    print(f"Episode Length: {results['episode_length']}")
    print(f"Total Distance: {results['total_distance']}")
    mins, secs = divmod(results["total_time"], 60)
    print(f"Total Time: {int(mins)} min/s, {secs:.2f} sec/s")
    print()

    return results