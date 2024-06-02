from tkinter import RIGHT
import airsim
import numpy as np
import math
import time
import pprint
from enum import Enum

# import gymnasium as gym
from gymnasium import spaces
from airsim import DrivetrainType, YawMode

from gym_drone.envs.drone_env_base import DroneEnv_Base

# from gymnasium.utils import seeding
    
class DiscAction(Enum):
    LEFT = 0
    FRONT_LEFT = 1
    FRONT = 2
    FRONT_RIGHT = 3
    RIGHT = 4
    BACK_RIGHT = 5
    BACK = 6
    BACK_LEFT = 7
    DOWN = 8
    UP = 9
    
class DroneEnvDisc(DroneEnv_Base):
    
    # Methods
    def __init__(self, env_config: dict):
        print("Drone Env Disc V2.5:")
        print(pprint.pformat(env_config))
        
        # Gym Variables
        # self.observation_space = spaces.Dict({
        #     "target_vector": spaces.Box(low=obs_norm.getint('min_pos'), high=obs_norm.getint('max_pos'), shape=(3,), dtype=np.float64),
        #     "relative_orientation": spaces.Box(low=obs_norm.getint('min_orient'), high=obs_norm.getint('max_orient'), shape=(1,), dtype=np.float64),
        #     "depth": spaces.Box(low=obs_norm.getint('min_depth'), high=obs_norm.getint('max_depth'), shape=env_config["image_shape"], dtype=np.uint8),
        #     "distances": spaces.Box(low=obs_norm.getint('min_dist'), high=obs_norm.getint('max_dist'), shape=(6,), dtype=np.float64),
        #     "collided": spaces.Box(low=0, high=1, shape=(1,), dtype=np.int8)
        # })
        self.observation_space = spaces.Box(low=-1, high=1, shape=(9,), dtype=np.float64)
        self.action_space = spaces.Discrete(10)
        
        super().__init__(env_config)
        
        self.observation_list = ["target_vector", "distances"]
    
    def _interprate_action(self, action):
        # (x_vel, y_vel, z_vel, yaw)
        if action == DiscAction.LEFT.value:
            quad_offset = (0.0, -self.SCALE_FACTOR, 0.0, -90.0)
        elif action == DiscAction.FRONT_LEFT.value:
            quad_offset = (self.SCALE_FACTOR, -self.SCALE_FACTOR, 0.0, -45.0)
        elif action == DiscAction.FRONT.value:
            quad_offset = (self.SCALE_FACTOR, 0.0, 0.0, 0.0)
        elif action == DiscAction.FRONT_RIGHT.value:
            quad_offset = (self.SCALE_FACTOR, self.SCALE_FACTOR, 0.0, 45.0)
        elif action == DiscAction.RIGHT.value:
            quad_offset = (0.0, 0.0, self.SCALE_FACTOR, 90.0)
        elif action == DiscAction.BACK_RIGHT.value:
            quad_offset = (-self.SCALE_FACTOR, self.SCALE_FACTOR, 0.0, 135.0)
        elif action == DiscAction.BACK.value:
            quad_offset = (-self.SCALE_FACTOR, 0.0, 0.0, 180.0)
        elif action == DiscAction.BACK_LEFT.value:
            quad_offset = (-self.SCALE_FACTOR, -self.SCALE_FACTOR, 0.0, -135.0)
        elif action == DiscAction.DOWN.value:
            yaw = self._get_attitude(deg=True)[2]
            quad_offset = (0.0, 0.0, self.SCALE_FACTOR, yaw)
        elif action == DiscAction.UP.value:
            yaw = self._get_attitude(deg=True)[2]
            quad_offset = (0.0, 0.0, -self.SCALE_FACTOR, yaw)
        else:
            raise ValueError(f"Invalid action: {action}")

        return quad_offset
    
    def _do_action(self, action):
        quad_offset = self._interprate_action(action)
        time.sleep(self.SLEEP_TIME)
        quad_vel = self._get_velocity()
        
        vel_x = quad_offset[0]
        vel_y = quad_offset[1]
        vel_z = quad_offset[2]
        
        if self.momentum:
            vel_x += quad_vel[0]
            vel_y += quad_vel[1]
            vel_z += quad_vel[2]
        
        self.drone.moveByVelocityAsync(
            vx          = vel_x,
            vy          = vel_y,
            vz          = vel_z,
            duration    = self.MOVEMENT_INTERVAL,
            drivetrain  = DrivetrainType.MaxDegreeOfFreedom,
            # yaw_mode    = YawMode(is_rate=False, yaw_or_rate=quad_offset[3])
            vehicle_name= self.drone_name
        ).join()
        