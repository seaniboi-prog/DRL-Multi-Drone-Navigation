import airsim
import time
import pickle

def get_xyz():
    """Function to get x, y, z coordinates from the user as a single comma-separated string."""
    try:
        coords = input("Enter x, y, z coordinates separated by commas: ")
        x, y, z = map(float, coords.split(','))
        return x, y, z
    except ValueError:
        print("Invalid input. Please enter numeric values separated by commas.")
        return get_xyz()
    
def rearrange_target(client: airsim.VehicleClient):
    fixed = False
    while not fixed:
        x, y, z = get_xyz()
        pose = airsim.Pose(airsim.Vector3r(x, y, -z), airsim.to_quaternion(0, 0, 0))
        client.simSetVehiclePose(pose, True)
        response = input(f"Is this the correct target? ({x}, {y}, {z}) (y/n): ")
        if response.lower() == 'y':
            fixed = True
    return [x, y, z]

def main():
    inp = input("Take from already generated waypoints? (y/n): ")
    generated = True if inp.lower() == 'y' else False
    
    # Connect to the AirSim simulator
    client = airsim.VehicleClient()
    client.confirmConnection()


    if generated:
        env_name = input("Enter the name of the environment: ")
        
        waypoints = pickle.load(open(f'{env_name}_waypoints_generated.pkl', 'rb'))
        valid_wps = []
        
        for waypoint in waypoints:
            x, y = waypoint
            x = float(x)
            y = float(y)
            z = 10.0
            pose = airsim.Pose(airsim.Vector3r(x, y, -z), airsim.to_quaternion(0, 0, 0))
            client.simSetVehiclePose(pose, True)
            print("Waypoint:", x, y, z)
            response = input("Is this a valid waypoint? (y/n): ")
            if response.lower() == 'y':
                valid_wps.append([x, y, z])
            else:
                valid_wps.append(rearrange_target(client))
                
        print("Valid Waypoints:")
        print(valid_wps)
        
        filename = f'{env_name}_waypoints_valid.pkl'
        with open(filename, 'wb') as outp:
            pickle.dump(valid_wps, outp, pickle.HIGHEST_PROTOCOL)
    
    else:
        try:    
            while True:
                # Get coordinates from the user
                x, y, z = get_xyz()

                # Teleport the drone to the specified coordinates
                pose = airsim.Pose(airsim.Vector3r(x, y, -z), airsim.to_quaternion(0, 0, 0))
                client.simSetVehiclePose(pose, True)
                
                # Pause for a moment before the next input
                time.sleep(1)

        except KeyboardInterrupt:
            print("\nExiting the program.")

        finally:
            # Land the drone and release controls
            client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(0, 0, 0)), True)

if __name__ == "__main__":
    main()
