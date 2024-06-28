import numpy as np

waypoints = dict()
obstacles = dict()

waypoints["blocks"] = [
                np.array([0.0, 0.0, 5.0], dtype=np.float32), # starting point
                np.array([75.0, 0.0, 10.0], dtype=np.float32),
                np.array([50.0, 60.0, 8.0], dtype=np.float32),
                np.array([50.0, -60.0, 12.0], dtype=np.float32),
                np.array([20.0, 110.0, 15.0], dtype=np.float32),
                np.array([20.0, -110.0, 10.0], dtype=np.float32),
                np.array([-30.0, 70.0, 6.0], dtype=np.float32),
                np.array([-30.0, -70.0, 16.0], dtype=np.float32),
                np.array([-50.0, 0.0, 5.0], dtype=np.float32),
                np.array([-60.0, 110.0, 20.0], dtype=np.float32),
                np.array([-60.0, -110.0, 13.0], dtype=np.float32),
                np.array([-100.0, 0.0, 7.0], dtype=np.float32),
                np.array([-115.0, 80.0, 11.0], dtype=np.float32),
                np.array([-115.0, -80.0, 18.0], dtype=np.float32),
            ]

obstacles["blocks"] = [
                np.array([19, -25, 0, 65, 22, 15]), # Obs 1
                np.array([27, 37, 0, 40, 40, 15]), # Obs 2
                np.array([10, 45, 0, 35, 70, 15]), # Obs 3
                np.array([40, 95, 0, 65, 120, 15]), # Obs 4
                np.array([-100, 95, 0, -75, 120, 15]), # Obs 5
                np.array([-80, 45, 0, -65, 60, 15]), # Obs 6
                np.array([-40, 12, 0, -15, 35, 15]), # Obs 7
                np.array([-40, -48, 0, -15, -25, 15]), # Obs 8
                np.array([-80, -63, 0, -65, -50, 15]), # Obs 9
                np.array([-100, -125, 0, -75, -100, 15]), # Obs 10
                np.array([10, -75, 0, 35, -48, 15]), # Obs 11
                np.array([40, -125, 0, 65, -100, 15]), # Obs 12
            ]

waypoints["neighbourhood"] = [

            ]

waypoints["africa"] = [

            ]

waypoints["mountains"] = [

            ]

waypoints["adrl"] = [

            ]


def get_waypoints(type: str) -> list:
    return waypoints[type]

def get_obstacles(type: str) -> list:
    return obstacles[type]