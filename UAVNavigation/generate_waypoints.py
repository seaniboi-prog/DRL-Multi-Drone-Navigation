from pickle_utils import *
import numpy as np
import random


def collide_with_obstacles( wypt_pos: list, obstacles: list):
    collided = False
    
    xd, yd = wypt_pos
    
    for obs in obstacles:
        x1, y1 = obs[:2]
        x2, y2 = obs[2:]

        # Check if the drone is within the bounds of the cuboid
        inside_x = x1 <= xd <= x2
        inside_y = y1 <= yd <= y2

        # Drone is inside the cuboid if all conditions are met
        collided = inside_x and inside_y
        
        if collided:
            break
        
    return collided

def generate_waypoints(x_range, y_range, obstacles, filename):

    random.seed(43)

    obs_free_waypoints = []

    for x in range(x_range[0], x_range[1]):
        for y in range(y_range[0], y_range[1]):
            if not collide_with_obstacles([x, y], obstacles):
                z = random.uniform(5, 20)
                obs_free_waypoints.append([x, y, z])            

    random.shuffle(obs_free_waypoints)

    print(obs_free_waypoints)
                
    # save_obj_file('waypoints/cust_blocks_waypoints.pkl', obs_free_waypoints)
    save_obj_file(filename, obs_free_waypoints)
    
    
obstacles_cust = []

obstacles_cust.append(np.array([20, -25, 65, 20])) # Obs 1
obstacles_cust.append(np.array([-10, -80, 40, -50])) # Obs 2
obstacles_cust.append(np.array([10, 45, 50, 65])) # Obs 3
obstacles_cust.append(np.array([-40, -25, -20, 20])) # Obs 4

# generate_waypoints([-150, 150], [-150, 150], obstacles_cust, 'gym_drone/envs/waypoints/cust_blocks_waypoints.pkl')


obstacles_blocks = []

obstacles_blocks.append(np.array([19, -25, 65, 22])) # Obs 1
obstacles_blocks.append(np.array([27, 37, 40, 40])) # Obs 2
obstacles_blocks.append(np.array([10, 45, 35, 70])) # Obs 3
obstacles_blocks.append(np.array([40, 95, 65, 120])) # Obs 4
obstacles_blocks.append(np.array([-100, 95, -75, 120])) # Obs 5
obstacles_blocks.append(np.array([-80, 45, -65, 60])) # Obs 6
obstacles_blocks.append(np.array([-40, 15, -15, 40])) # Obs 7
obstacles_blocks.append(np.array([-40, -45, -15, 20])) # Obs 8
obstacles_blocks.append(np.array([-80, -63, -65, -50])) # Obs 9
obstacles_blocks.append(np.array([-100, -125, -75, -100])) # Obs 10
obstacles_blocks.append(np.array([10, -75, 35, -48])) # Obs 11
obstacles_blocks.append(np.array([40, -125, 65, -100])) # Obs 12


generate_waypoints([-150, 150], [-150, 150], obstacles_blocks, 'gym_drone/envs/waypoints/blocks_waypoints.pkl')