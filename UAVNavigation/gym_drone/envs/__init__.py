try:
    from gym_drone.envs.drone_env_base import *
    from gym_drone.envs.drone_env_cust import *
    from gym_drone.envs.drone_env_disc import *
    from gym_drone.envs.drone_env_cont import *
except ImportError:
    from UAVNavigation.gym_drone.envs.drone_env_base import *
    from UAVNavigation.gym_drone.envs.drone_env_cust import *
    from UAVNavigation.gym_drone.envs.drone_env_disc import *
    from UAVNavigation.gym_drone.envs.drone_env_cont import *