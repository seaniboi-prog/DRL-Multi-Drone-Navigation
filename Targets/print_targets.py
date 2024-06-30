import pickle

env_name = input("Enter the name of the environment: ")
        
waypoints = pickle.load(open(f'{env_name}_waypoints_valid.pkl', 'rb'))
valid_wps = []

for waypoint in waypoints:
    x, y, z = waypoint
    
    print(f"np.array([{x}, {y}, {z}], dtype=np.float32),")
    
print("Total waypoints:", len(waypoints))