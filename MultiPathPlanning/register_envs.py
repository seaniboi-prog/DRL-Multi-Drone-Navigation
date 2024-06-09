from gymnasium.envs.registration import register

from ray.tune.registry import register_env

from UAVNavigation.gym_drone.envs import *
from MultiPathPlanning.coordinates import get_waypoints

def register_ray_gym_envs(*args, **kwargs):
    
    waypoint_type = kwargs.get('waypoint_type')
    
    end_at_start = kwargs.get('end_at_start', False)

    env_config = {
        "waypoints": get_waypoints(waypoint_type),
        "max_steps": None,
        "render_mode": None,
        "verbose": False,
        "end_at_start": end_at_start
    }
    
    # Register Custom Environments

    register(
        id="drone-env-disc-cust",
        entry_point="UAVNavigation.gym_drone.envs:DroneEnvCust_Disc",
        kwargs={"env_config": env_config},
    )
    register_env("drone-env-disc-cust", lambda config: DroneEnvCust_Disc(env_config=config))

    register(
        id="drone-env-cont-cust",
        entry_point="UAVNavigation.gym_drone.envs:DroneEnvCust_Cont",
        kwargs={"env_config": env_config},
    )
    register_env("drone-env-cont-cust", lambda config: DroneEnvCust_Cont(env_config=config))

    # Register Discrete Environments

    register(
        id="drone-env-disc-airsim",
        entry_point="UAVNavigation.gym_drone.envs:DroneEnvDisc",
        kwargs={"env_config": env_config},
    )
    register_env("drone-env-disc-airsim", lambda config: DroneEnvDisc(env_config=config))

    # Register Continuous Environments

    register(
        id="drone-env-cont-airsim",
        entry_point="UAVNavigation.gym_drone.envs:DroneEnvCont",
        kwargs={"env_config": env_config},
    )
    register_env("drone-env-cont-airsim", lambda config: DroneEnvCont(env_config=config))
