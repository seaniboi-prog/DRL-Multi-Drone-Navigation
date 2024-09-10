from gymnasium.envs.registration import register
from gymnasium.wrappers.time_limit import TimeLimit

from ray.tune.registry import register_env

try:
    from gym_drone.envs import *
    from gym_drone.waypoint_utils import get_waypoints
except ImportError:
    from UAVNavigation.gym_drone.envs import *
    from UAVNavigation.gym_drone.waypoint_utils import get_waypoints

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
            "far_limit": 50,
            "render_mode": render_mode,
            "verbose": verbose,
            "momentum": momentum,
            "random_waypts": num_rand_waypts,
    }
    
    if num_rand_waypts == 0:
        env_config["waypoints"] = get_waypoints(waypoint_type)
        
    if "exp_waypts" in kwargs and kwargs["exp_waypts"] is not None:
        env_config["waypoints"] = kwargs["exp_waypts"]
        
    if "end_at_start" in kwargs and kwargs["end_at_start"] is not None:
        env_config["end_at_start"] = kwargs["end_at_start"]

    env_ids = []

    # Register Custom Environments

    register(
        id="drone-env-disc-cust",
        entry_point="gym_drone.envs:DroneEnvCust_Disc",
        kwargs={"env_config": env_config},
    )
    register_env("drone-env-disc-cust", lambda config: DroneEnvCust_Disc(env_config=config))
    env_ids.append("drone-env-disc-cust")

    register(
        id="drone-env-cont-cust",
        entry_point="gym_drone.envs:DroneEnvCust_Cont",
        kwargs={"env_config": env_config},
    )
    register_env("drone-env-cont-cust", lambda config: DroneEnvCust_Cont(env_config=config))
    env_ids.append("drone-env-cont-cust")

    # Register Discrete Environments

    register(
        id="drone-env-disc-airsim",
        entry_point="gym_drone.envs:DroneEnvDisc",
        kwargs={"env_config": env_config},
    )
    register_env("drone-env-disc-airsim", lambda config: DroneEnvDisc(env_config=config))
    env_ids.append("drone-env-disc-airsim")

    # Register Continuous Environments

    register(
        id="drone-env-cont-airsim",
        entry_point="gym_drone.envs:DroneEnvCont",
        kwargs={"env_config": env_config},
    )
    register_env("drone-env-cont-airsim", lambda config: DroneEnvCont(env_config=config))
    env_ids.append("drone-env-cont-airsim")

    return env_config, env_ids