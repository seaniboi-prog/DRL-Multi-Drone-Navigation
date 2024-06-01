from  gym_drone.envs import *

from gymnasium.envs.registration import register
from gymnasium.wrappers.time_limit import TimeLimit

from ray.tune.registry import register_env

from gym_drone.waypoint_utils import get_waypoints

def get_env_config(*args, **kwargs):
    
    waypoint_type = kwargs.get('waypoint_type', "single")
    
    random_waypts = kwargs.get('random_waypts', True)
    
    momentum = kwargs.get('momentum', False)
    
    if random_waypts:
        if waypoint_type == "single":
            num_rand_waypts = 1
        elif waypoint_type == "multiple":
            num_rand_waypts = 5
        else:
            raise ValueError("Invalid waypoint_type for random waypoints")
    else:
        num_rand_waypts = 0
        
    if "max_steps" in kwargs and kwargs["max_steps"] is not None:
        max_steps = kwargs["max_steps"]
    else:
        if waypoint_type == "single":
            max_steps = 300
        elif waypoint_type == "multiple":
            max_steps = 500
        else:
            raise ValueError("Invalid waypoint_type for max_steps")
    
    verbose = kwargs.get('verbose', False)
    
    render_mode = kwargs.get('render_mode', None)

    env_config = {
            "max_steps": max_steps,
            "image_shape": (84, 84, 1),
            "away_limit": 30,
            "far_limit": 50,
            "render_mode": render_mode,
            "verbose": verbose,
            "momentum": momentum,
            "random_waypts": num_rand_waypts,
    }
    
    if num_rand_waypts == 0:
        env_config["waypoints"] = get_waypoints(waypoint_type)

    env_ids = []

    # Register Custom Environments

    register(
        id="drone-env-cust-v1",
        entry_point="gym_drone.envs:DroneEnvCust_v1",
        kwargs={"env_config": env_config},
        max_episode_steps=max_steps
    )
    register_env("drone-env-cust-v1", lambda config: TimeLimit(DroneEnvCust_v1(env_config=config), max_episode_steps=max_steps))
    env_ids.append("drone-env-cust-v1")

    register(
        id="drone-env-cust-v2",
        entry_point="gym_drone.envs:DroneEnvCust_v2",
        kwargs={"env_config": env_config},
        max_episode_steps=max_steps
    )
    register_env("drone-env-cust-v2", lambda config: TimeLimit(DroneEnvCust_v2(env_config=config), max_episode_steps=max_steps))
    env_ids.append("drone-env-cust-v2")

    # Register Discrete Environments

    register(
        id="drone-env-disc-v1",
        entry_point="gym_drone.envs:DroneEnvDisc_v1",
        kwargs={"env_config": env_config},
        max_episode_steps=max_steps
    )
    register_env("drone-env-disc-v1", lambda config: TimeLimit(DroneEnvDisc_v1(env_config=config), max_episode_steps=max_steps))
    env_ids.append("drone-env-disc-v1")

    register(
        id="drone-env-disc-v2",
        entry_point="gym_drone.envs:DroneEnvDisc_v2",
        kwargs={"env_config": env_config},
        max_episode_steps=max_steps
    )
    register_env("drone-env-disc-v2", lambda config: TimeLimit(DroneEnvDisc_v2(env_config=config), max_episode_steps=max_steps))
    env_ids.append("drone-env-disc-v2")

    register(
        id="drone-env-disc-v3",
        entry_point="gym_drone.envs:DroneEnvDisc_v3",
        kwargs={"env_config": env_config},
        max_episode_steps=max_steps
    )
    register_env("drone-env-disc-v3", lambda config: TimeLimit(DroneEnvDisc_v3(env_config=config), max_episode_steps=max_steps))
    env_ids.append("drone-env-disc-v3")

    register(
        id="drone-env-disc-v4",
        entry_point="gym_drone.envs:DroneEnvDisc_v4",
        kwargs={"env_config": env_config},
        max_episode_steps=max_steps
    )
    register_env("drone-env-disc-v4", lambda config: TimeLimit(DroneEnvDisc_v4(env_config=config), max_episode_steps=max_steps))
    env_ids.append("drone-env-disc-v4")

    # Register Continuous Environments

    register(
        id="drone-env-cont-v1",
        entry_point="gym_drone.envs:DroneEnvCont_v1",
        kwargs={"env_config": env_config},
        max_episode_steps=max_steps
    )
    register_env("drone-env-cont-v1", lambda config: TimeLimit(DroneEnvCont_v1(env_config=config), max_episode_steps=max_steps))
    env_ids.append("drone-env-cont-v1")

    register(
        id="drone-env-cont-v2",
        entry_point="gym_drone.envs:DroneEnvCont_v2",
        kwargs={"env_config": env_config},
        max_episode_steps=max_steps
    )
    register_env("drone-env-cont-v2", lambda config: TimeLimit(DroneEnvCont_v2(env_config=config), max_episode_steps=max_steps))
    env_ids.append("drone-env-cont-v2")

    return env_config, env_ids