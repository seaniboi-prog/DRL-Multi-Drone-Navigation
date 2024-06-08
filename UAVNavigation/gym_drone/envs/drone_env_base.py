import gymnasium as gym
import airsim
import numpy as np
import time
import math
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
import copy
import os
import pickle

from airsim import Vector3r, ImageType

try:
    from gym_drone.envs.env_utils import *
except ImportError:
    from UAVNavigation.gym_drone.envs.env_utils import *

class DroneEnv_Base(gym.Env):
    # Constants
    SLEEP_TIME = 0.0001
    SCALE_FACTOR = 1.0
    MOVEMENT_INTERVAL = 1.0
    CKPT_THRESH = 3.0
    OBS_THRESH = 3.0
    DEFAULT_CAM = "front"
    
    # Rewards
    REWARD_CRASH = -200
    REWARD_FAR_AWAY = -20
    REWARD_AWAY = -10
    REWARD_HOVER = -1
    REWARD_MULTIPLIER = 4
    REWARD_CLOSER = 5
    REWARD_CKPT = 100
    REWARD_GOAL = 200
    REWARD_FUNC_RANGE = 1
    
    # Methods
    def __init__(self, env_config: dict):
        # Custom Variables
        self.max_steps = env_config.get("max_steps", 500)
        self.image_shape = env_config.get("image_shape", (84, 84, 1))
        self.image_type = env_config.get("image_type", ImageType.DepthPerspective)
        self.random_waypts = env_config.get("random_waypts", 0)
        self.end_at_start = env_config.get("end_at_start", False)
        if self.random_waypts == 0:
            self.waypoints: list = env_config["waypoints"]
            if self.end_at_start:
                self.waypoints.append(self._get_position())
            self.goal_idx = len(self.waypoints) - 1
        self.verbose = env_config.get("verbose", False)
        self.far_limit = env_config.get("far_limit", 50)
        self.momentum = env_config.get("momentum", True)
        self.render_mode = env_config.get("render_mode", None)
        self.drone_name = env_config.get("drone_name", "Drone1")
        
        self.reward_range = (self.REWARD_CRASH, self.REWARD_GOAL)
        self.episode_count = 0
        self.timestep = 0
        self.totalsteps = 0
        self.extra_steps = 0
        self.waypt_idx = 0
        self.start_time = time.time()
        self.camera = self.DEFAULT_CAM
        self.observation_list = []
        self.route = []
        
        waypoint_root = os.path.join("C:\\Users\\seanf\\Documents\\Workspace\\DRL-Multi-Drone-Navigation\\UAVNavigation\\gym_drone\\envs", "waypoints", "blocks_waypoints.pkl")
        self.rand_waypt_choices = pickle.load(open(waypoint_root, "rb"))
        
        # Starting Simulation API
        self.drone = airsim.MultirotorClient()
        self.drone.confirmConnection()
        
        self.reset()
        
    def reset(self, seed=None, options=None):
        self.episode_count += 1
        # print("Episode {}: with {} steps".format(self.episode_count, self.timestep))
        self.drone.simPrintLogMessage(f"Starting New Episode {self.episode_count}")
        if self.random_waypts > 0:
            # self.waypoints = get_random_coors([5, 10], [15, 45], z_range=[3., 15.], count=self.episode_count, num_coors=self.random_waypts, env_variant="blocks")
            self.waypoints = random.sample(self.rand_waypt_choices, self.random_waypts)
            
            if self.end_at_start:
                self.waypoints.append(self._get_position())
            
            self.goal_idx = len(self.waypoints) - 1
                
        self.waypt_idx = 0
        self.timestep = 0
        self.extra_steps = 0
        self.start_time = time.time()
        self.state = dict()
        
        if options is not None and "_reset" in options:
            self._prepare_takeoff(reset=options["_reset"])
        else:
            self._prepare_takeoff()
        
        time.sleep(self.SLEEP_TIME)
        curr_pos = self._get_position()
        self.state["position"] = curr_pos
        self.last_dist = self._get_distance(curr_pos, self.waypoints[self.waypt_idx])
        self.route = [copy.deepcopy(curr_pos)]
        self._update_state()

        return self._get_obs(), self.state
    
    def _prepare_takeoff(self, reset=True):
        if reset:
            self.drone.reset()
        self.drone.enableApiControl(True, vehicle_name=self.drone_name)
        self.drone.armDisarm(True, vehicle_name=self.drone_name)
        self.drone.takeoffAsync(vehicle_name=self.drone_name).join()
        
    def _get_position(self) -> np.ndarray:
        pos: Vector3r = self.drone.getMultirotorState(vehicle_name=self.drone_name).kinematics_estimated.position
        return np.array([pos.x_val, pos.y_val, -pos.z_val])
    
    def _get_velocity(self) -> np.ndarray:
        vel: Vector3r = self.drone.getMultirotorState(vehicle_name=self.drone_name).kinematics_estimated.linear_velocity
        return np.array([vel.x_val, vel.y_val, vel.z_val])
    
    def _get_distance(self, quad_pos: np.ndarray, next_waypoint: np.ndarray) -> float:
        return float(abs(np.linalg.norm(quad_pos - next_waypoint)))
        
    def _get_target_vector(self, quad_pos: np.ndarray, next_waypoint: np.ndarray) -> np.ndarray:
        x_dist = next_waypoint[0] - quad_pos[0]
        y_dist = next_waypoint[1] - quad_pos[1]
        z_dist = next_waypoint[2] - quad_pos[2]
            
        return np.array([x_dist, y_dist, z_dist])
    
    def _get_route(self):
        return np.array(self.route)
    
    def _get_path_dist(self) -> float:
        return float(np.sum([abs(np.linalg.norm(self.route[i] - self.route[i+1])) for i in range(len(self.route) - 1)]))
    
    def _update_state(self):
        time.sleep(self.SLEEP_TIME)
        
        self.state["prev_position"] = self.state["position"]
        self.state["position"] = self._get_position()
        self.state["lin_velocity"] = self._get_velocity()
        self.state["orientation"] = self._get_attitude(deg=True)
        self.state["collision"] = self._drone_collision(False)
        self.state["progress"] = self.waypt_idx / len(self.waypoints)
        self.state["route"] = self.route
        self.state["distance_travelled"] = self._get_path_dist()
        self.state["time_elapsed"] = time.time() - self.start_time
        self.state["status"] = "running"
        
    def _normalize_obs(self, obs: np.ndarray, d_min: int, d_max: int, cust_min: int, cust_max: int) -> np.ndarray:
        norm_obs = cust_min + (obs - d_min) * (cust_max - cust_min) / (d_max - d_min)
        return np.clip(norm_obs, cust_min, cust_max)
        
    def _get_relative_yaw(self, curr_pos: np.ndarray, target_pos: np.ndarray, curr_attitude: list):
        relative_pose_x = target_pos[0] - curr_pos[0]
        relative_pose_y = target_pos[1] - curr_pos[1]
        angle = math.atan2(relative_pose_y, relative_pose_x)

        # get current yaw
        yaw_current = curr_attitude[2]

        # get yaw error
        yaw_error = angle - yaw_current
        
        return self._normalize_angle(yaw_error)
    
    def _get_depth_image(self):
        response = self.drone.simGetImages(
            [airsim.ImageRequest(self.camera, self.image_type, True, False)],
            vehicle_name=self.drone_name
        )[0]
        
        img1d = np.array(response.image_data_float)
        img1d = 255 / np.maximum(np.ones(img1d.size), img1d)
        img2d = img1d.reshape(response.height, response.width)
        
        image = Image.fromarray(img2d)
        img_w, img_h, img_c = self.image_shape
        im_resize = np.array(image.resize((img_w, img_h)).convert("L"))
        
        return im_resize.reshape([img_w, img_h, img_c])
    
    def _drone_collision(self, num: bool):
        collision_info = self.drone.simGetCollisionInfo(vehicle_name=self.drone_name)
        
        if collision_info.has_collided or collision_info.object_name != "" or collision_info.object_id != -1:
            collided = True
        elif self._get_position()[2] < -0.5:
            collided = True
        else:
            collided = False
        
        if num:
            return 1 if collided else 0
        else:
            return collided
        
    def _get_obs(self):
        time.sleep(self.SLEEP_TIME)
        
        # Retrieve position
        pos = self._get_position()
        self.route.append(copy.deepcopy(pos))
        
        observations = np.array([])
        
        if "target_vector" in self.observation_list:
            target_vector = self._get_target_vector(pos, self.waypoints[self.waypt_idx])
            unit_target_vector = unit_vector(target_vector)
            observations = np.concatenate((observations, unit_target_vector))
            
        # if "relative_orientation" in self.observation_list:
        #     curr_attitude = self._get_attitude(deg=False)
        #     relative_yaw = self._get_relative_yaw(pos, self.waypoints[self.waypt_idx], curr_attitude)
        #     relative_orientation = np.array([relative_yaw])
        #     norm_rel_orient = self._normalize_obs(relative_orientation, -180, 180, -1, 1)
        #     observations = np.concatenate((observations, norm_rel_orient))
            
        if "distances" in self.observation_list:
            distances = self._get_sensor_distances()
            norm_dists = self._normalize_obs(distances, 0, 10, 0, 1)
            observations = np.concatenate((observations, norm_dists))
        
        return observations
    
    # def _get_obs_old(self):
    #     time.sleep(self.SLEEP_TIME)
        
    #     observation = {}
    #     pos = self._get_position()
    #     self.route.append(copy.deepcopy(pos))
        
    #     # Retrieve position
    #     if "target_vector" in self.observation_list:
    #         target_vector = self._get_target_vector(pos, self.waypoints[self.waypt_idx])
    #         unit_target_vector = unit_vector(target_vector)
    #         observation["target_vector"] = unit_target_vector
        
    #     # Retrieve orientation
    #     if "relative_orientation" in self.observation_list:
    #         curr_attitude = self._get_attitude(deg=False)
    #         relative_yaw = self._get_relative_yaw(pos, self.waypoints[self.waypt_idx], curr_attitude)
    #         relative_orientation = np.array([relative_yaw])
    #         norm_rel_orient = self._normalize_obs(relative_orientation, -180, 180, -1, 1)
    #         observation["relative_orientation"] = norm_rel_orient
        
    #     # Calculate Distances
    #     if "distances" in self.observation_list:
    #         distances = self._get_sensor_distances()
    #         norm_dists = self._normalize_obs(distances, 0, 10, 0, 1)
    #         observation["distances"] = norm_dists
        
    #     # Retrieve image
    #     if "depth" in self.observation_list:
    #         flat_image = self._get_depth_image().flatten()
    #         flat_norm = self._normalize_obs(flat_image, 0, 255, 0, 1)
    #         normalized_image = flat_norm.reshape(self.image_shape)
    #         observation["depth"] = normalized_image
        
    #     # Collided
    #     if "collided" in self.observation_list:
    #         observation["collided"] = np.array([self._drone_collision(True)])
        
    #     return observation
    
    def step(self, action):
        self._do_action(action)
        self._update_state()
        self.state["action"] = action
        obs = self._get_obs()
        
        if self.max_steps is not None:
            truncated = self.timestep >= (self.max_steps + self.extra_steps)
        else:
            truncated = False
        self.timestep += 1
        self.totalsteps += 1
        
        if self.verbose and self.timestep % 10 == 0:
            pos_str = ', '.join([str(round(i, 1)) for i in self._get_position()])
            goal_str = ', '.join([str(round(i, 1)) for i in self.waypoints[self.waypt_idx]])
            self.drone.simPrintLogMessage(f"Episode: {self.episode_count} | Step: {self.timestep} | Pos: {pos_str} | Goal: {goal_str} | Total Steps: {self.totalsteps}")
        
        reward, terminated = self._determine_reward()

        if truncated:
            self.drone.simPrintLogMessage(f"{self.drone_name} - Episode {self.episode_count}: TIMED OUT", severity=2)
            self.state["status"] = "timed_out"
        
        return obs, reward, terminated, truncated, self.state
    
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
        
    def _get_attitude(self, deg) -> list:
        drone_orientation = self.drone.getMultirotorState(vehicle_name=self.drone_name).kinematics_estimated.orientation
        drone_attitude = airsim.to_eularian_angles(drone_orientation)
        
        return [self._normalize_angle(angle, deg) for angle in drone_attitude]
    
    def _get_sensor_distances(self) -> np.ndarray:
        # Calculate Distances
        dist_sensors = ["front", "right", "bottom", "rear", "left", "top"]
        suf = "_distance"
        distances = np.array([self.drone.getDistanceSensorData(dist+suf, vehicle_name=self.drone_name).distance for dist in dist_sensors])
        clipped = np.clip(distances, 0, 10)
        return clipped
    
    def _get_min_sensor_distance(self):
        return np.min(self._get_sensor_distances())

    def _do_action(self, action):
        raise NotImplementedError()
    
    def _determine_reward(self):
        if self._drone_collision(False):
            reward = self.REWARD_CRASH
            self.drone.simPrintLogMessage(f"{self.drone_name} - Episode {self.episode_count}: CRASHED", severity=2)
            self.state["status"] = "crashed"
            terminated = True
        else:
            curr_pos = self._get_position()
            dist = self._get_distance(curr_pos, self.waypoints[self.waypt_idx])
            
            if dist <= self.CKPT_THRESH:
                # check if next waypoint is goal
                if self.waypt_idx == self.goal_idx:
                    reward = self.REWARD_GOAL
                    self.drone.simPrintLogMessage(f"{self.drone_name} - Episode {self.episode_count}: SOLVED", severity=1)
                    self.state["progress"] = 1.0
                    self.state["status"] = "solved"
                    terminated = True
                else:
                    reward = self.REWARD_CKPT
                    self.drone.simPrintLogMessage(f"{self.drone_name} - Episode {self.episode_count}: CHECKPOINT {self.waypt_idx + 1}/{len(self.waypoints)} REACHED", severity=3)
                    terminated = False
                    self.waypt_idx += 1
                    self.extra_steps = copy.copy(self.timestep)
                    self.last_dist = self._get_distance(curr_pos, self.waypoints[self.waypt_idx])
                    self.state["progress"] = self.waypt_idx / len(self.waypoints)
            
            else:
                reward = 0
                terminated = False
                
                diff = self.last_dist - dist
                self.last_dist = dist
                
                reward += self._calculate_reward(diff, var="negative")
                
                if self.verbose and self.timestep % 10 == 0:
                    self.drone.simPrintLogMessage(f"Episode {self.episode_count}: Reward Function - Dist: {dist:.2f} | Diff: {diff:.2f} | Reward: {reward:.2f}")

                # min_dist = self._get_min_sensor_distance()

                # if min_dist < self.OBS_THRESH:
                #     reward -= 1 - np.clip((min_dist / self.OBS_THRESH), 0,1)
                
                # if "relative_orientation" in self.observation_list:
                #     relative_yaw = self._get_relative_yaw(curr_pos,
                #                                         self.waypoints[self.waypt_idx],
                #                                         self._get_attitude(deg=False))
                #     pos_rad_rel_yaw = abs(math.radians(relative_yaw))
                    
                #     reward -= pos_rad_rel_yaw
                
        return reward, terminated
    
    def _calculate_reward(self, diff, var="balanced"):

        if var == "balanced":
            if diff > 0:
                calc_rew = self.REWARD_MULTIPLIER * ( diff ** 4 ) / (self.REWARD_FUNC_RANGE ** 4 )
                return min(calc_rew, self.REWARD_CLOSER)
            elif diff < 0:
                calc_rew = -1 * self.REWARD_MULTIPLIER * ( diff ** 2 ) / ( (self.REWARD_FUNC_RANGE ** 2) / 2)
                return max(calc_rew, self.REWARD_AWAY)
            else:
                return 0.0
            
        elif var == "negative":
            if diff > 0:
                calc_rew = 2 * ( diff ** 4 ) / (self.REWARD_FUNC_RANGE ** 4 )
            else:
                calc_rew = -1 * self.REWARD_MULTIPLIER * ( diff ** 2 ) / ( (self.REWARD_FUNC_RANGE ** 2) / 2)
            
            return np.clip(calc_rew - 2, -12, 0)
        
        elif var == "diff":
            return diff
        
        else:
            raise ValueError(f"Invalid reward variant: {var}")
        
    def plot_reward_shaping(self, var="balanced"):
        x_limit = self.REWARD_FUNC_RANGE
        
        x_p = np.linspace(0, x_limit, 100)
        y_p = [self._calculate_reward(i, var) for i in x_p]
        
        x_n = np.linspace(-x_limit, 0, 100)
        y_n = [self._calculate_reward(i, var) for i in x_n]
        
        z = np.zeros(int(x_limit * 2 + 1))
        
        plt.figure(figsize=(8, 8))
        plt.plot(x_p, y_p, color='lime', zorder=2)
        plt.plot(x_n, y_n, color='red', zorder=2)
        plt.plot(np.arange(-x_limit, x_limit + 1), z, color='blue', linestyle='--', zorder=1)
        plt.xlabel('Distance Difference')
        plt.ylabel('Reward')
        plt.title('Reward Shaping')
        plt.show()
        
    def render(self):
        if self.render_mode is None:
            return
        elif self.render_mode == "plot":
            self._live_plot_route()
        elif self.render_mode == "rgb_array":
            response = self.drone.simGetImages(
                [airsim.ImageRequest(self.DEFAULT_CAM, ImageType.Scene, False, False)],
                vehicle_name=self.drone_name
            )[0]
            
            image_data = np.array(airsim.string_to_uint8_array(response.image_data_uint8))

            # Reshape the data into an image (height, width, channels)
            image = image_data.reshape(response.height, response.width, 3)
            return image
        
    def _plot_path(self, live=True):
        if not live:
            plt.figure(figsize=(8, 8))
        
        curr_pos = self._get_position()
        
        # Plot the current position
        plt.scatter(curr_pos[0], curr_pos[1], c='b', s=30, marker=MarkerStyle(marker='o'))
            
        # Plot route
        route = np.array(self.route)
        plt.plot(route[:, 0], route[:, 1], c='green')
        
        # Plot the waypoints
        waypoints = np.array(self.waypoints)
        if self.waypt_idx == self.goal_idx:
            plt.scatter(waypoints[:self.waypt_idx, 0], waypoints[:self.waypt_idx, 1], c='orange', s=50, marker=MarkerStyle('x'))
            plt.scatter(waypoints[self.waypt_idx, 0], waypoints[self.waypt_idx, 1], c='red', s=50, marker=MarkerStyle('x'))
        else:
            plt.scatter(waypoints[:self.waypt_idx + 1, 0], waypoints[:self.waypt_idx + 1, 1], c='orange', s=50, marker=MarkerStyle('x'))
        plt.text(-90, 90, f"Episode: {(self.episode_count - 1)}", fontsize=12)
        
        # Plot the z axis
        if live:
            zx_values = [95, 95]
            zy_values = [75, 95]
            plt.plot(zx_values, zy_values, c='k', zorder=1)
            plt.scatter(zx_values[0], zy_values[0] + self.waypoints[self.waypt_idx][2], c='red', s=50, marker=MarkerStyle('x'), zorder=2)
            plt.scatter(zx_values[0], zy_values[0] + curr_pos[2], c='b', marker=MarkerStyle('o'), zorder=2)
            plt.text(92, 72, "z axis", fontsize=10)
        
        # Set the limits and labels
        plt.xlim([-100, 100])
        plt.ylim([-100, 100])
        plt.xlabel('x axis')
        plt.ylabel('y axis')
        
        if not live:
            plt.show()
    
    def _live_plot_route(self):
        # Clear plot if already exists
        if not plt.get_fignums():
            plt.figure(figsize=(8, 8))
        else:
            plt.clf()

        curr_pos = self._get_position()
        
        # Plot the current position
        plt.scatter(curr_pos[0], curr_pos[1], c='b', s=30, marker=MarkerStyle(marker='o'))
        if self.waypt_idx != self.goal_idx:
            w_col = 'orange'
        else:
            w_col = 'red'
            
        # Plot route
        route = np.array(self.route)
        plt.plot(route[:, 0], route[:, 1], c='green')
        
        # Plot the waypoints
        plt.scatter(self.waypoints[:self.waypt_idx][0], self.waypoints[:self.waypt_idx][1], c=w_col, s=50, marker=MarkerStyle('x'))
        plt.text(-90, 90, f"Episode: {(self.episode_count - 1)}", fontsize=12)
        
        # Plot the z axis
        zx_values = [95, 95]
        zy_values = [75, 95]
        plt.plot(zx_values, zy_values, c='k', zorder=1)
        plt.scatter(zx_values[0], zy_values[0] + self.waypoints[self.waypt_idx][2], c='red', s=50, marker=MarkerStyle('x'), zorder=2)
        plt.scatter(zx_values[0], zy_values[0] + curr_pos[2], c='b', marker=MarkerStyle('o'), zorder=2)
        plt.text(92, 72, "z axis", fontsize=10)
        
        # Set the limits and labels
        plt.xlim([-100, 100])
        plt.ylim([-100, 100])
        plt.xlabel('x axis')
        plt.ylabel('y axis')
        plt.pause(0.05)
    
    def close(self):
        self.drone.enableApiControl(False, vehicle_name=self.drone_name)
        self.drone.armDisarm(False, vehicle_name=self.drone_name)
        print('Disconnected!')

