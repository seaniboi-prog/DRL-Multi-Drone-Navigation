import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import Rectangle
from matplotlib.markers import MarkerStyle

def plot_obstacles(obstacles, waypoints=[], xlim=[-150, 150], ylim=[-150, 150]):
    plt.figure(figsize=(8, 8))

    # Plot the obstacles
    for obs in obstacles:
        x_min, y_min, _, x_max, y_max, _ = obs
        plt.gca().add_patch(Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, fill=True, color='grey', alpha=0.8, zorder=1))

    # Plot the waypoints
    if len(waypoints) > 0:

        plt.scatter(waypoints[0][0], waypoints[0][1], c='green', s=50, marker=MarkerStyle('o'))

        for wp in waypoints[1:]:
            x, y, _ = wp
            plt.scatter(x, y, c='red', s=50, marker=MarkerStyle('x'))

    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel('x axis')
    plt.ylabel('y axis')

    plt.show()

# Obstacles [x_min, y_min, z_min, x_max, y_max, z_max]

# Block Obstacles
block_obstacles = []
block_obstacles.append(np.array([19, -25, 0, 65, 22, 15])) # Obs 1
block_obstacles.append(np.array([27, 37, 0, 40, 40, 15])) # Obs 2
block_obstacles.append(np.array([10, 45, 0, 35, 70, 15])) # Obs 3
block_obstacles.append(np.array([40, 95, 0, 65, 120, 15])) # Obs 4
block_obstacles.append(np.array([-100, 95, 0, -75, 120, 15])) # Obs 5
block_obstacles.append(np.array([-80, 45, 0, -65, 60, 15])) # Obs 6
block_obstacles.append(np.array([-40, 12, 0, -15, 35, 15])) # Obs 7
block_obstacles.append(np.array([-40, -48, 0, -15, -25, 15])) # Obs 8
block_obstacles.append(np.array([-80, -63, 0, -65, -50, 15])) # Obs 9
block_obstacles.append(np.array([-100, -125, 0, -75, -100, 15])) # Obs 10
block_obstacles.append(np.array([10, -75, 0, 35, -48, 15])) # Obs 11
block_obstacles.append(np.array([40, -125, 0, 65, -100, 15])) # Obs 12

# Block Waypoints
block_waypoints = []
block_waypoints.append(np.array([0.0, 0.0, 5.0], dtype=np.float32)) # starting point
block_waypoints.append(np.array([75.0, 0.0, 10.0], dtype=np.float32))
block_waypoints.append(np.array([50.0, 60.0, 8.0], dtype=np.float32))
block_waypoints.append(np.array([50.0, -60.0, 12.0], dtype=np.float32))
block_waypoints.append(np.array([20.0, 110.0, 15.0], dtype=np.float32))
block_waypoints.append(np.array([20.0, -110.0, 10.0], dtype=np.float32))
block_waypoints.append(np.array([-30.0, 70.0, 6.0], dtype=np.float32))
block_waypoints.append(np.array([-30.0, -70.0, 16.0], dtype=np.float32))
block_waypoints.append(np.array([-50.0, 0.0, 5.0], dtype=np.float32))
block_waypoints.append(np.array([-60.0, 110.0, 20.0], dtype=np.float32))
block_waypoints.append(np.array([-60.0, -110.0, 13.0], dtype=np.float32))
block_waypoints.append(np.array([-100.0, 0.0, 7.0], dtype=np.float32))
block_waypoints.append(np.array([-115.0, 80.0, 11.0], dtype=np.float32))
block_waypoints.append(np.array([-115.0, -80.0, 18.0], dtype=np.float32))



plot_obstacles(block_obstacles, block_waypoints)