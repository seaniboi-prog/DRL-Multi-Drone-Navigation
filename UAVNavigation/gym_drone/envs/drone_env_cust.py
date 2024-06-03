import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
from enum import Enum
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
from matplotlib.patches import Rectangle
import pprint
import copy
import pickle
import random
import time
import os

from typing import Union

# from utils import *
from gym_drone.envs.env_utils import *

class CustActionDisc(Enum):
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

class DroneEnvCust_Disc(gym.Env):
    
    # Constants
    SCALE_FACTOR = 1.0
    CKPT_THRESH = 3.0
    
    # Rewards
    REWARD_CRASH = -200
    REWARD_FAR_AWAY = -20
    REWARD_AWAY = -10
    REWARD_HOVER = -1
    REWARD_MULTIPLIER = 5
    REWARD_CLOSER = 5
    REWARD_CKPT = 100
    REWARD_GOAL = 300

    def __init__(self, env_config: dict):
        print("Drone Env Cust Discrete:")
        print(pprint.pformat(env_config))
        
        # Custom Variables
        self.max_steps = env_config.get("max_steps", 200)
        self.random_waypts = env_config.get("random_waypts", 0)
        self.end_at_start = env_config.get("end_at_start", False)
        if self.random_waypts == 0:
            self.waypoints: list = env_config["waypoints"]
            if self.end_at_start:
                self.waypoints.append(copy.copy(self.curr_pos))
            self.goal_idx = len(self.waypoints) - 1
        self.verbose = env_config.get("verbose", False)
        self.render_mode = env_config.get("render_mode", None)
        
        self.reward_range = (self.REWARD_CRASH, self.REWARD_GOAL)
        self.episode_count = 0
        self.timestep = 0
        self.curr_pos = np.array([0.0, 0.0, 3.0])
        self.curr_orient = 0
        self.start_time = time.time()
        self.route = []
        self.x_range = [-200, 200]
        self.y_range = [-200, 200]
        self.z_range = [0, 20]
        
        # Obstacles [x_min, y_min, z_min, x_max, y_max, z_max]
        self.obstacles = []
        self.obstacles.append(np.array([19, -25, 0, 65, 22, 15])) # Obs 1
        self.obstacles.append(np.array([27, 37, 0, 40, 40, 15])) # Obs 2
        self.obstacles.append(np.array([10, 45, 0, 35, 70, 15])) # Obs 3
        self.obstacles.append(np.array([40, 95, 0, 65, 120, 15])) # Obs 4
        self.obstacles.append(np.array([-100, 95, 0, -75, 120, 15])) # Obs 5
        self.obstacles.append(np.array([-80, 45, 0, -65, 60, 15])) # Obs 6
        self.obstacles.append(np.array([-40, 15, 0, -15, 40, 15])) # Obs 7
        self.obstacles.append(np.array([-40, -45, 0, -15, 20, 15])) # Obs 8
        self.obstacles.append(np.array([-80, -63, 0, -65, -50, 15])) # Obs 9
        self.obstacles.append(np.array([-100, -125, 0, -75, -100, 15])) # Obs 10
        self.obstacles.append(np.array([10, -75, 0, 35, -48, 15])) # Obs 11
        self.obstacles.append(np.array([40, -125, 0, 65, -100, 15])) # Obs 12
        
        waypoint_root = os.path.join("C:\\Users\\seanf\\Documents\\Workspace\\DRL-Multi-Drone-Navigation\\UAVNavigation\\gym_drone\\envs", "waypoints", "blocks_waypoints.pkl")
        self.rand_waypt_choices = pickle.load(open(waypoint_root, "rb"))
                
        # Gym Variables
        # self.observation_space = spaces.Dict({
        #     "position": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64),
        #     "target_vector": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64),
        #     "relative_orientation": spaces.Box(low=-180, high=180, shape=(1,), dtype=np.float64),
        #     "collided": spaces.Box(low=0, high=1, shape=(1,), dtype=np.int8)
        # })
        self.observation_space = spaces.Box(low=-1, high=1, shape=(9,), dtype=np.float64)

        self.action_space = spaces.Discrete(10)
        
        self.reset()

    def reset(self, *args, **kwargs):
        self.episode_count += 1
        # print("Episode {}: with {} steps".format(self.episode_count, self.timestep))
        if self.random_waypts > 0:
            random.seed(self.episode_count)
            # self.waypoints = get_random_coors(5., 145., count=self.episode_count, num_coors=self.random_waypts)
            self.waypoints = random.sample(self.rand_waypt_choices, self.random_waypts)
            
            if self.end_at_start:
                self.waypoints.append(copy.copy(self.curr_pos))
            
            self.goal_idx = len(self.waypoints) - 1
            
        self.waypt_idx = 0
        self.timestep = 0
        self.away_count = 0
        self.curr_pos = np.array([0.0, 0.0, 3.0])
        self.route = [copy.deepcopy(self.curr_pos)]
        self.curr_orient = 0
        self.state = dict()
        self.start_time = time.time()
                
        self.state["position"] = self.curr_pos
        self.last_dist = self._get_distance(self.curr_pos, self.waypoints[self.waypt_idx])
        self._update_state()

        return self._get_obs(), self.state
    
    def _get_route(self):
        return np.array(self.route)
    
    def _get_path_dist(self) -> float:
        return float(np.sum([abs(np.linalg.norm(self.route[i] - self.route[i+1])) for i in range(len(self.route) - 1)]))
    
    def _normalize_angle(self, angle, deg=True, reverse=False):
        if reverse:
            # keep angle between 0 and 2pi
            ang_rad = (angle + 2 * math.pi) % (2 * math.pi)
        else:
            # keep angle between -pi and pi
            ang_rad = (angle + np.pi) % (2 * np.pi) - np.pi

        if deg:
            return math.degrees(ang_rad)
        else:
            return ang_rad
    
    def _get_distance(self, quad_pos: np.ndarray, next_waypoint: np.ndarray) -> float:
        
        return float(abs(np.linalg.norm(quad_pos - next_waypoint)))
        
    def _get_target_vector(self, quad_pos: np.ndarray, next_waypoint) -> np.ndarray:

        x_dist = next_waypoint[0] - quad_pos[0]
        y_dist = next_waypoint[1] - quad_pos[1]
        z_dist = next_waypoint[2] - quad_pos[2]
            
        return np.array([x_dist, y_dist, z_dist])
    
    def _update_state(self):
        
        self.state["prev_position"] = self.state["position"]
        self.state["position"] = self.curr_pos
        self.state["orientation"] = self.curr_orient
        self.state["progress"] = self.waypt_idx / len(self.waypoints)
        self.state["route"] = self.route
        self.state["distance_travelled"] = self._get_path_dist()
        self.state["time_elapsed"] = time.time() - self.start_time
        self.state['solved'] = False

    def _get_yaw(self, deg) -> float:
        return self._normalize_angle(self.curr_orient, deg)
    
    def _get_relative_yaw(self, curr_pos: np.ndarray, target_pos: np.ndarray, curr_yaw: float):
        relative_pose_x = target_pos[0]- curr_pos[0]
        relative_pose_y = target_pos[1] - curr_pos[1]
        angle = math.atan2(relative_pose_y, relative_pose_x)

        # get current yaw
        yaw_current = curr_yaw

        # get yaw error
        yaw_error = angle - yaw_current
        
        return self._normalize_angle(yaw_error)
    
    def _get_sensor_dists_outer(self, direction):
        if direction == "x_pos":
            return self.x_range[1] - self.curr_pos[0]
        elif direction == "x_neg":
            return self.curr_pos[0] - self.x_range[0]
        elif direction == "y_pos":
            return self.y_range[1] - self.curr_pos[1]
        elif direction == "y_neg":
            return self.curr_pos[1] - self.y_range[0]
        elif direction == "z_pos":
            return self.z_range[1] - self.curr_pos[2]
        elif direction == "z_neg":
            return self.curr_pos[2] - self.z_range[0]
        else:
            raise ValueError(f"Invalid direction: {direction}")
        
    def _get_sensor_dists_obstacles(self, direction, obstacle):
        x_min, y_min, z_min, x_max, y_max, z_max = obstacle
        
        curr_x, curr_y, curr_z = self.curr_pos
        
        if self._collide_with_obstacles():
            return 0.0
        
        if "x" in direction:
            if y_min <= curr_y <= y_max and z_min <= curr_z <= z_max:
                if direction == "x_pos":
                    if curr_x < x_min:
                        return x_min - curr_x
                    else:
                        return self._get_sensor_dists_outer(direction)
                else:
                    if curr_x > x_max:
                        return curr_x - x_max
                    else:
                        return self._get_sensor_dists_outer(direction)
            else:
                return self._get_sensor_dists_outer(direction)
            
        elif "y" in direction:
            if x_min <= curr_x <= x_max and z_min <= curr_z <= z_max:
                if direction == "y_pos":
                    if curr_y < y_min:
                        return y_min - curr_y
                    else:
                        return self._get_sensor_dists_outer(direction)
                else:
                    if curr_y > y_max:
                        return curr_y - y_max
                    else:
                        return self._get_sensor_dists_outer(direction)
            else:
                return self._get_sensor_dists_outer(direction)
            
        elif "z" in direction:
            if x_min <= curr_x <= x_max and y_min <= curr_y <= y_max:
                if direction == "z_pos":
                    if curr_z < z_min:
                        return z_min - curr_z
                    else:
                        return self._get_sensor_dists_outer(direction)
                else:
                    if curr_z > z_max:
                        return curr_z - z_max
                    else:
                        return self._get_sensor_dists_outer(direction)
            else:
                return self._get_sensor_dists_outer(direction)
            
        else:
            raise ValueError(f"Invalid direction: {direction}")
            
    
    def _get_sensor_dists(self):
        dist_dirs = ["x_pos", "x_neg", "y_pos", "y_neg", "z_pos", "z_neg"]
        
        if self._collide_with_obstacles():
            return np.zeros(len(dist_dirs))
        
        global_dists = np.full(len(dist_dirs), np.inf)
        
        for obstacle in self.obstacles:
            local_dists = []
            for direction in dist_dirs:
                dist = self._get_sensor_dists_obstacles(direction, obstacle)
                local_dists.append(dist)
            
            global_dists = np.minimum(global_dists, np.array(local_dists))
            
        return global_dists

    def _get_obs(self):
        
        target_vector = self._get_target_vector(self.curr_pos, self.waypoints[self.waypt_idx])
        unit_target_vector = unit_vector(target_vector)
        
        sensor_dists = self._get_sensor_dists()
        norm_sensor_dists = normalize_arr(sensor_dists)
        # print(norm_sensor_dists)
        
        return np.concatenate((unit_target_vector, norm_sensor_dists))
        
        # Retrieve position
        # position = self._get_target_vector(self.curr_pos, self.waypoints[self.waypt_idx])
        
        # Retrieve orientation
        # curr_yaw = self._get_yaw(deg=False)
        # relative_yaw = self._get_relative_yaw(self.curr_pos, self.waypoints[self.waypt_idx], curr_yaw)
        # relative_orientation = np.array([relative_yaw])

        # collided = 1 if self._did_crash() else 0
        
        # observation = {
        #     "position": np.array(self.curr_pos),
        #     "target_vector": position,
        #     "relative_orientation": relative_orientation,
        #     "collided": np.array([collided])
        # }
        
        # return observation

    def step(self, action):
        self._do_action(action)
        self._update_state()
        self.state["action"] = action
        obs = self._get_obs()
        
        self.route.append(copy.deepcopy(self.curr_pos))
        
        if self.max_steps is not None:
            truncated = self.timestep >= self.max_steps
        else:
            truncated = False
        self.timestep += 1
        
        reward, terminated = self._determine_reward()
        
        self.render()
        
        return obs, reward, terminated, truncated, self.state
    
    def _interprate_action(self, action):
        # (x_vel, y_vel, z_vel, yaw)
        if action == CustActionDisc.LEFT.value:
            quad_offset = (0.0, -self.SCALE_FACTOR, 0.0, -90.0)
        elif action == CustActionDisc.FRONT_LEFT.value:
            quad_offset = (self.SCALE_FACTOR, -self.SCALE_FACTOR, 0.0, -45.0)
        elif action == CustActionDisc.FRONT.value:
            quad_offset = (self.SCALE_FACTOR, 0.0, 0.0, 0.0)
        elif action == CustActionDisc.FRONT_RIGHT.value:
            quad_offset = (self.SCALE_FACTOR, self.SCALE_FACTOR, 0.0, 45.0)
        elif action == CustActionDisc.RIGHT.value:
            quad_offset = (0.0, 0.0, self.SCALE_FACTOR, 90.0)
        elif action == CustActionDisc.BACK_RIGHT.value:
            quad_offset = (-self.SCALE_FACTOR, self.SCALE_FACTOR, 0.0, 135.0)
        elif action == CustActionDisc.BACK.value:
            quad_offset = (-self.SCALE_FACTOR, 0.0, 0.0, 180.0)
        elif action == CustActionDisc.BACK_LEFT.value:
            quad_offset = (-self.SCALE_FACTOR, -self.SCALE_FACTOR, 0.0, -135.0)
        elif action == CustActionDisc.DOWN.value:
            quad_offset = (0.0, 0.0, self.SCALE_FACTOR, self.curr_orient)
        elif action == CustActionDisc.UP.value:
            quad_offset = (0.0, 0.0, -self.SCALE_FACTOR, self.curr_orient)
        else:
            raise ValueError(f"Invalid action: {action}")

        return quad_offset
    
    def _do_action(self, action):
        quad_offset = self._interprate_action(action)
        
        self.curr_pos += np.array(quad_offset[:3])
        self.curr_orient += quad_offset[3]

    def _did_crash(self):
        if self.x_range[0] > self.curr_pos[0] or self.curr_pos[0] > self.x_range[1]:
            return True
        elif self.y_range[0] > self.curr_pos[1] or self.curr_pos[1] > self.y_range[1]:
            return True
        elif self.z_range[0] > self.curr_pos[2] or self.curr_pos[2] > self.z_range[1]:
            return True
        elif self._collide_with_obstacles():
            return True
        else:
            return False
        
    def _collide_with_obstacles(self, drone_pos: Union[list, None] = None):
        collided = False
        if drone_pos is None:
            xd, yd, zd = self.curr_pos
        else:
            xd, yd, zd = drone_pos
        
        for obs in self.obstacles:
            x1, y1, z1 = obs[:3]
            x2, y2, z2 = obs[3:]

            # Check if the drone is within the bounds of the cuboid
            inside_x = x1 <= xd <= x2
            inside_y = y1 <= yd <= y2
            inside_z = z1 <= zd <= z2

            # Drone is inside the cuboid if all conditions are met
            collided = inside_x and inside_y and inside_z
            
            if collided:
                break
            
        return collided
    
    def _determine_reward(self):
        if self._did_crash():
            if self.verbose:
                print("CRASHED")
            reward = self.REWARD_CRASH
            terminated = True
        else:
            dist = self._get_distance(self.curr_pos, self.waypoints[self.waypt_idx])
            
            if dist <= self.CKPT_THRESH:
                # check if next waypoint is goal
                if self.waypt_idx == self.goal_idx:
                    reward = self.REWARD_GOAL
                    if self.verbose:
                        print("SOLVED")
                    terminated = True
                    self.state['progress'] = 1.0
                    self.state['solved'] = True
                else:
                    reward = self.REWARD_CKPT
                    terminated = False
                    self.waypt_idx += 1
                    self.state['progress'] = self.waypt_idx / len(self.waypoints)
                    self.last_dist = self._get_distance(self.curr_pos, self.waypoints[self.waypt_idx])
            
            else:
                reward = 0
                terminated = False
                
                if self.verbose:
                    print(f"Previous distance to next waypoint: {round(self.last_dist, 2)}")
                    print(f"Distance to next waypoint: {round(dist, 2)}")
                
                diff = self.last_dist - dist
                self.last_dist = dist
                
                if self.verbose:
                    print(f"Distance difference: {round(diff, 2)}")
                
                reward += self._calculate_reward(diff, var="negative")
                
                if self.verbose:
                    print(f"Distance Reward: {round(reward, 2)}")
                
        return reward, terminated
    

    def _calculate_reward(self, diff, var="balanced"):
        if var == "balanced":
            if diff > 0:
                calc_rew = self.REWARD_MULTIPLIER * ( diff ** 4 ) / (self.SCALE_FACTOR ** 4)
                return min(calc_rew, self.REWARD_CLOSER)
            elif diff < 0:
                calc_rew = -1 * self.REWARD_MULTIPLIER * ( diff ** 2 ) / (self.SCALE_FACTOR ** 2 / 2)
                return max(calc_rew, self.REWARD_AWAY)
            else:
                return 0.0
            
        elif var == "negative":
            if diff > 0:
                calc_rew = 2 * ( diff ** 4 ) / (self.SCALE_FACTOR ** 4)
            else:
                calc_rew = -1 * self.REWARD_MULTIPLIER * ( diff ** 2 ) / (self.SCALE_FACTOR ** 2 / 2)
            
            return np.clip(calc_rew - 2, -12, 0)
        
        elif var == "diff":
            return diff
        
        else:
            raise ValueError(f"Invalid reward variant: {var}")

    def render(self):
        if self.render_mode is None:
            return
        elif self.render_mode == "plot":
            self._live_plot()
            
    def _live_plot(self):
        # Clear plot if already exists
        if not plt.get_fignums():
            plt.figure(figsize=(8, 8))
        else:
            plt.clf()
        
        # Plot the current position
        plt.scatter(self.curr_pos[0], self.curr_pos[1], c='b', s=30, marker=MarkerStyle(marker='o'), zorder=3)
            
        # Plot route
        route = np.array(self.route)
        plt.plot(route[:, 0], route[:, 1], c='green', zorder=2)
        
        # Plot the waypoints
        waypoints = np.array(self.waypoints)
        if self.waypt_idx == self.goal_idx:
            plt.scatter(waypoints[:self.waypt_idx, 0], waypoints[:self.waypt_idx, 1], c='orange', s=50, marker=MarkerStyle('x'))
            plt.scatter(waypoints[self.waypt_idx, 0], waypoints[self.waypt_idx, 1], c='red', s=50, marker=MarkerStyle('x'))
        else:
            plt.scatter(waypoints[:self.waypt_idx + 1, 0], waypoints[:self.waypt_idx + 1, 1], c='orange', s=50, marker=MarkerStyle('x'))
        plt.text(-140, 140, f"Episode: {(self.episode_count - 1)}", fontsize=12)
        
        # Plot the obstacles
        for obs in self.obstacles:
            x_min, y_min, _, x_max, y_max, _ = obs
            plt.gca().add_patch(Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, fill=True, color='grey', alpha=0.8, zorder=1))
        
        # Plot the z axis
        zx_values = [145, 145]
        zy_values = [125, 145]
        plt.plot(zx_values, zy_values, c='k', zorder=1)
        plt.scatter(zx_values[0], zy_values[0] + self.waypoints[self.waypt_idx][2], c='red', s=50, marker=MarkerStyle('x'), zorder=2)
        plt.scatter(zx_values[0], zy_values[0] + self.curr_pos[2], c='b', marker=MarkerStyle('o'), zorder=3)
        plt.scatter(zx_values[0], zy_values[0] + 15, c='grey', marker=MarkerStyle('^'), zorder=2)
        plt.text(142, 122, "z axis", fontsize=10)
        
        # Set the limits and labels
        plt.xlim(self.x_range)
        plt.ylim(self.y_range)
        plt.xlabel('x axis')
        plt.ylabel('y axis')
        plt.pause(0.05)

    def close(self):
        print('Closed!')
        if self.render_mode == "plot":
            plt.close()

class DroneEnvCust_Cont(DroneEnvCust_Disc):
    def __init__(self, env_config: dict):
        print("Drone Env Cust Continuous:")
        print(pprint.pformat(env_config))
        
        # Custom Variables
        self.max_steps = env_config.get("max_steps", 200)
        self.random_waypts = env_config.get("random_waypts", 0)
        self.end_at_start = env_config.get("end_at_start", False)
        if self.random_waypts == 0:
            self.waypoints = env_config["waypoints"]
            if self.end_at_start:
                self.waypoints.append(copy.copy(self.curr_pos))
            self.goal_idx = len(self.waypoints) - 1
        self.verbose = env_config.get("verbose", False)
        self.render_mode = env_config.get("render_mode", None)
        self.reward_range = (self.REWARD_CRASH, self.REWARD_GOAL)
        self.episode_count = 0
        self.timestep = 0
        self.curr_pos = np.array([0.0, 0.0, 3.0])
        self.curr_orient = 0
        self.route = []
        self.start_time = time.time()

        if env_config.get("end_at_start", False):
            self.waypoints.append(self._get_position())

        self.x_range = [-200, 200]
        self.y_range = [-200, 200]
        self.z_range = [0, 20]
        
        # Obstacles [x_min, y_min, z_min, x_max, y_max, z_max]
        self.obstacles = []
        self.obstacles.append(np.array([19, -25, 0, 65, 22, 15])) # Obs 1
        self.obstacles.append(np.array([27, 37, 0, 40, 40, 15])) # Obs 2
        self.obstacles.append(np.array([10, 45, 0, 35, 70, 15])) # Obs 3
        self.obstacles.append(np.array([40, 95, 0, 65, 120, 15])) # Obs 4
        self.obstacles.append(np.array([-100, 95, 0, -75, 120, 15])) # Obs 5
        self.obstacles.append(np.array([-80, 45, 0, -65, 60, 15])) # Obs 6
        self.obstacles.append(np.array([-40, 15, 0, -15, 40, 15])) # Obs 7
        self.obstacles.append(np.array([-40, -45, 0, -15, 20, 15])) # Obs 8
        self.obstacles.append(np.array([-80, -63, 0, -65, -50, 15])) # Obs 9
        self.obstacles.append(np.array([-100, -125, 0, -75, -100, 15])) # Obs 10
        self.obstacles.append(np.array([10, -75, 0, 35, -48, 15])) # Obs 11
        self.obstacles.append(np.array([40, -125, 0, 65, -100, 15])) # Obs 12
        
        waypoint_root = os.path.join("C:\\Users\\seanf\\Documents\\Workspace\\DRL-Multi-Drone-Navigation\\UAVNavigation\\gym_drone\\envs", "waypoints", "blocks_waypoints.pkl")
        self.rand_waypt_choices = pickle.load(open(waypoint_root, "rb"))

        # Gym Variables
        # self.observation_space = spaces.Dict({
        #     "position": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64),
        #     "target_vector": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64),
        #     "relative_orientation": spaces.Box(low=-180, high=180, shape=(1,), dtype=np.float64),
        #     "collided": spaces.Box(low=0, high=1, shape=(1,), dtype=np.int8)
        # })
        self.observation_space = spaces.Box(low=-1, high=1, shape=(9,), dtype=np.float64)

        self.action_space = spaces.Box(low=-self.SCALE_FACTOR, high=self.SCALE_FACTOR, shape=(3,), dtype=np.float64)
        
        self.reset()

    def _do_action(self, action):
        self.curr_pos += action