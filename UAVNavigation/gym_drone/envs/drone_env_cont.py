import airsim
import numpy as np
import math
import time
import pprint

from gymnasium import spaces
from airsim import DrivetrainType, YawMode

try:
    from gym_drone.envs.drone_env_base import DroneEnv_Base
except ImportError:
    from UAVNavigation.gym_drone.envs.drone_env_base import DroneEnv_Base

class DroneEnvCont(DroneEnv_Base):
    
    # Methods
    def __init__(self, env_config: dict):
        print("Drone Env Cont V1:")
        print(pprint.pformat(env_config))
        
        # Gym Variables
        # self.observation_space = spaces.Dict({
        #     "target_vector": spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float64),
        #     "relative_orientation": spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float64),
        #     "depth": spaces.Box(low=0, high=1, shape=env_config["image_shape"], dtype=np.uint8),
        #     "distances": spaces.Box(low=0, high=1, shape=(6,), dtype=np.float64),
        #     "collided": spaces.Box(low=0, high=1, shape=(1,), dtype=np.int8)
        # })
        
        self.observation_space = spaces.Box(low=-1, high=1, shape=(9,), dtype=np.float64)
        
        self.action_space = spaces.Box(low=-self.SCALE_FACTOR, high=self.SCALE_FACTOR,
                                       shape=(3,), dtype=np.float64)
        
        super().__init__(env_config)
        
        self.observation_list = ["target_vector", "distances"]
    
    def _calculate_yaw_from_velocity(self, vx, vy, deg):
        yaw = math.atan2(vy, vx)
        return math.degrees(yaw) if deg else yaw
    
    def _calculate_yaw_rate_from_yaw(self, desired_yaw, duration):
        current_yaw = self._get_attitude(deg=False)[2]
        diff_yaw = abs(desired_yaw - current_yaw)
        normalized_diff_yaw = self._normalize_angle(diff_yaw, deg=True)
        
        yaw_rate = normalized_diff_yaw / duration
            
        return yaw_rate
    
    def _do_action(self, action):
        time.sleep(self.SLEEP_TIME)
        quad_vel = self._get_velocity()

        # Calculate velocities
        vx = action[0]
        vy = action[1]
        vz = action[2]
        
        if self.momentum:
            vx += quad_vel[0]
            vy += quad_vel[1]
            vz += quad_vel[2]

        # Calculate yaw rate
        # desired_yaw = self._calculate_yaw_from_velocity(vx, vy, deg=False)
        # pos_desired_yaw = self._normalize_angle(desired_yaw, deg=False, reverse=True)
                
        # yaw_rate = self._calculate_yaw_rate_from_yaw(pos_desired_yaw, self.MOVEMENT_INTERVAL)

        self.drone.moveByVelocityAsync(
            vx          = vx,
            vy          = vy,
            vz          = vz,
            duration    = self.MOVEMENT_INTERVAL,
            drivetrain  = DrivetrainType.MaxDegreeOfFreedom,
            # yaw_mode    = YawMode(is_rate=True, yaw_or_rate=yaw_rate),
            vehicle_name = self.drone_name
        ).join()