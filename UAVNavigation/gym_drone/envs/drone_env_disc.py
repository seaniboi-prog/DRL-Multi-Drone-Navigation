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


class DiscActionV1(Enum):
    HOVER = 0
    FRONT = 1
    RIGHT = 2
    DOWN = 3
    BACK = 4
    LEFT = 5
    UP = 6

class DroneEnvDisc_v1(DroneEnv_Base):
    
    # Methods
    def __init__(self, env_config: dict):
        print("Drone Env Disc V1:")
        print(pprint.pformat(env_config))
        
        # Gym Variables
        self.observation_space = spaces.Dict({
            "target_vector": spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float64),
            "depth": spaces.Box(low=0, high=1, shape=env_config["image_shape"], dtype=np.uint8),
            "distances": spaces.Box(low=0, high=1, shape=(6,), dtype=np.float64),
            "collided": spaces.Box(low=0, high=1, shape=(1,), dtype=np.int8)
        })
        self.action_space = spaces.Discrete(7)
        
        super().__init__(env_config)
        
        self.observation_list = ["target_vector", "depth", "distances", "collided"]
    
    def _get_camera(self):
        action = self.state.get("action", DiscActionV1.FRONT.value)
        
        if action == DiscActionV1.HOVER.value:
            camera_name = self.DEFAULT_CAM
        elif action == DiscActionV1.FRONT.value:
            camera_name = "front"
        elif action == DiscActionV1.RIGHT.value:
            camera_name = "right"
        elif action == DiscActionV1.DOWN.value:
            camera_name = "front"
        elif action == DiscActionV1.BACK.value:
            camera_name = "back"
        elif action == DiscActionV1.LEFT.value:
            camera_name = "left"
        elif action == DiscActionV1.UP.value:
            camera_name = "front"
        else:
            camera_name = self.DEFAULT_CAM

        return camera_name
    
    def _interprate_action(self, action):
        if action == DiscActionV1.HOVER.value:
            quad_offset = (0, 0, 0)
        elif action == DiscActionV1.FRONT.value:
            quad_offset = (self.SCALE_FACTOR, 0, 0)
        elif action == DiscActionV1.RIGHT.value:
            quad_offset = (0, self.SCALE_FACTOR, 0)
        elif action == DiscActionV1.DOWN.value:
            quad_offset = (0, 0, self.SCALE_FACTOR)
        elif action == DiscActionV1.BACK.value:
            quad_offset = (-self.SCALE_FACTOR, 0, 0)
        elif action == DiscActionV1.LEFT.value:
            quad_offset = (0, -self.SCALE_FACTOR, 0)
        elif action == DiscActionV1.UP.value:
            quad_offset = (0, 0, -self.SCALE_FACTOR)
        else:
            raise ValueError(f"Invalid action: {action}")

        return quad_offset
    
    def _do_action(self, action):
        self.camera = self._get_camera()
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
            vehicle_name= self.drone_name
        ).join()
    
class DiscActionV2(Enum):
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
    
class DroneEnvDisc_v2(DroneEnv_Base):
    
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
        if action == DiscActionV2.LEFT.value:
            quad_offset = (0.0, -self.SCALE_FACTOR, 0.0, -90.0)
        elif action == DiscActionV2.FRONT_LEFT.value:
            quad_offset = (self.SCALE_FACTOR, -self.SCALE_FACTOR, 0.0, -45.0)
        elif action == DiscActionV2.FRONT.value:
            quad_offset = (self.SCALE_FACTOR, 0.0, 0.0, 0.0)
        elif action == DiscActionV2.FRONT_RIGHT.value:
            quad_offset = (self.SCALE_FACTOR, self.SCALE_FACTOR, 0.0, 45.0)
        elif action == DiscActionV2.RIGHT.value:
            quad_offset = (0.0, 0.0, self.SCALE_FACTOR, 90.0)
        elif action == DiscActionV2.BACK_RIGHT.value:
            quad_offset = (-self.SCALE_FACTOR, self.SCALE_FACTOR, 0.0, 135.0)
        elif action == DiscActionV2.BACK.value:
            quad_offset = (-self.SCALE_FACTOR, 0.0, 0.0, 180.0)
        elif action == DiscActionV2.BACK_LEFT.value:
            quad_offset = (-self.SCALE_FACTOR, -self.SCALE_FACTOR, 0.0, -135.0)
        elif action == DiscActionV2.DOWN.value:
            yaw = self._get_attitude(deg=True)[2]
            quad_offset = (0.0, 0.0, self.SCALE_FACTOR, yaw)
        elif action == DiscActionV2.UP.value:
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
    

class DiscActionV3(Enum):
    FRONT = 0
    YAW_RIGHT = 1
    YAW_LEFT = 2
    DOWN = 3
    UP = 4

class DroneEnvDisc_v3(DroneEnv_Base):
    
    # Methods
    def __init__(self, env_config: dict):
        print("Drone Env Disc V3:")
        print(pprint.pformat(env_config))
        
        # Gym Variables
        self.observation_space = spaces.Dict({
            "target_vector": spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float64),
            "relative_orientation": spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float64),
            "depth": spaces.Box(low=0, high=1, shape=env_config["image_shape"], dtype=np.uint8),
            "distances": spaces.Box(low=0, high=1, shape=(6,), dtype=np.float64),
            "collided": spaces.Box(low=0, high=1, shape=(1,), dtype=np.int8)
        })
        self.action_space = spaces.Discrete(5)
        
        super().__init__(env_config)
        
        self.observation_list = ["target_vector", "relative_orientation", "depth", "distances", "collided"]
    
    def _do_action(self, action):
        time.sleep(self.SLEEP_TIME)
        
        if action == DiscActionV3.FRONT.value:
            self._go_straight()
        elif action == DiscActionV3.YAW_RIGHT.value:
            self._yaw_right()
        elif action == DiscActionV3.YAW_LEFT.value:
            self._yaw_left()
        elif action == DiscActionV3.DOWN.value:
            self._go_vertical(False)
        elif action == DiscActionV3.UP.value:
            self._go_vertical(True)
    
    def _go_straight(self):
        quad_vel = self._get_velocity()
        
        x_vel = self.SCALE_FACTOR
        y_vel = 0
        z_vel = 0
        
        if self.momentum:
            x_vel += quad_vel[0]
            y_vel += quad_vel[1]
            z_vel += quad_vel[2]
        
        self.drone.moveByVelocityAsync(
            x_vel,
            y_vel,
            z_vel,
            self.MOVEMENT_INTERVAL,
            vehicle_name=self.drone_name
        ).join()
    
    def _go_vertical(self, up: bool):
        quad_vel = self._get_velocity()
        
        x_vel = 0
        y_vel = 0
        z_vel= -self.SCALE_FACTOR if up else self.SCALE_FACTOR
        
        if self.momentum:
            x_vel += quad_vel[0]
            y_vel += quad_vel[1]
            z_vel += quad_vel[2]
        
        self.drone.moveByVelocityAsync(
            x_vel,
            y_vel,
            z_vel,
            self.MOVEMENT_INTERVAL,
            vehicle_name=self.drone_name
        ).join()
    
    def _yaw_right(self):
        # turn 30 degrees to the right
        self.drone.rotateByYawRateAsync(90, 1/3, vehicle_name=self.drone_name).join() # 90 degrees/sec for 1/3 second

    def _yaw_left(self):
        # turn 30 degrees to the left
        self.drone.rotateByYawRateAsync(-90, 1/3, vehicle_name=self.drone_name).join() # 90 degrees/sec for 1/3 second
    
class DroneEnvDisc_v4(DroneEnvDisc_v3):
    # Methods
    def __init__(self, env_config: dict):
        print("Drone Env Disc V4:")
        print(pprint.pformat(env_config))
        
        # Gym Variables
        self.observation_space = spaces.Dict({
            "target_vector": spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float64),
            "relative_orientation": spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float64),
            "depth": spaces.Box(low=0, high=1, shape=env_config["image_shape"], dtype=np.uint8),
            "distances": spaces.Box(low=0, high=1, shape=(6,), dtype=np.float64),
            "collided": spaces.Box(low=0, high=1, shape=(1,), dtype=np.int8)
        })
        self.action_space = spaces.Discrete(5)
        
        super().__init__(env_config)
        
        self.observation_list = ["target_vector", "relative_orientation", "depth", "distances", "collided"]
        
    def _go_straight(self):
        quad_vel = self._get_velocity()
        
        yaw = self._get_attitude(deg=False)[2]
        
        # Calculate the velocity components in x and y directions
        vx = self.SCALE_FACTOR * math.cos(yaw)
        vy = self.SCALE_FACTOR * math.sin(yaw)
        vz = 0
        
        if self.momentum:
            vx += quad_vel[0]
            vy += quad_vel[1]
            vz += quad_vel[2]
        
        self.drone.moveByVelocityAsync(
            vx,
            vy,
            vz,
            self.MOVEMENT_INTERVAL,
            vehicle_name=self.drone_name
        ).join()