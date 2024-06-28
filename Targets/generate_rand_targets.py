import numpy as np
import matplotlib.pyplot as plt
import pickle

def generate_random_waypoints_old(x_min, y_min, x_max, y_max, num_waypoints):
    # Generate random points within the specified range
    x_points = np.random.uniform(x_min, x_max, num_waypoints)
    y_points = np.random.uniform(y_min, y_max, num_waypoints)
    
    # Combine x and y points into waypoints
    waypoints = np.column_stack((x_points, y_points))
    
    return waypoints

def generate_random_waypoints(x_min, y_min, x_max, y_max, num_waypoints, min_distance):
    waypoints = []
    
    while len(waypoints) < num_waypoints:
        # Generate a random point within the specified range
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)
        new_point = np.array([int(x), int(y)])
        
        # Check the distance to all existing waypoints
        if all(np.linalg.norm(new_point - wp) >= min_distance for wp in waypoints):
            waypoints.append(new_point)
    
    return np.array(waypoints)

def plot_waypoints(waypoints, x_min, y_min, x_max, y_max):
    plt.figure(figsize=(8, 8))
    plt.scatter(waypoints[:, 0], waypoints[:, 1], c='blue', marker='o')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Generated Waypoints')
    # plt.grid(True)
    plt.show()

def main():
    # Define area boundaries and number of waypoints
    x_min, y_min, x_max, y_max = -250, -250, 250, 250
    num_waypoints = 30
    min_dist = 50
    env_name = input("Enter the name of the environment: ")
    valid = False
    
    while not valid:
        # Generate waypoints
        waypoints = generate_random_waypoints(x_min, y_min, x_max, y_max, num_waypoints, min_dist)
        
        # Plot the waypoints
        plot_waypoints(waypoints, x_min, y_min, x_max, y_max)
        response = input("Are you satisfied with the waypoints? (y/n): ")
        valid = True if response.lower() == 'y' else False
    
    print("Generated Waypoints:")
    print(waypoints)
    
    filename = f'{env_name}_waypoints_generated.pkl'
    
    with open(filename, 'wb') as outp:
        pickle.dump(waypoints, outp, pickle.HIGHEST_PROTOCOL)

# Entry point of the script
if __name__ == "__main__":
    main()
