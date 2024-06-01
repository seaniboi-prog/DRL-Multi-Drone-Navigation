import numpy as np

waypoints = dict()

waypoints["far"] = [
                np.array([7.0, -27.0, 7.0], dtype=np.float32),
                np.array([60.0, -30.0, 6.0], dtype=np.float32),
                np.array([75.0, -10.0, 5], dtype=np.float32),
            ]

waypoints["multiple"] = [
                np.array([4.0, -10.0, 3.0], dtype=np.float32),
                np.array([8.0, -20.0, 6.0], dtype=np.float32),
                np.array([15.0, -10.0, 4.0], dtype=np.float32),
                np.array([9.0, -22.0, 5.0], dtype=np.float32),
                np.array([15.0, -30.0, 7.0], dtype=np.float32),
                np.array([26.0, -25.0, 5.0], dtype=np.float32),
            ]

waypoints["single"] = [
                np.array([9.0, -16.0, 5.0], dtype=np.float32)
            ]

waypoints["cust"] = [
                np.array([14.0, -10.0, 3.0], dtype=np.float32),
                np.array([-18.0, -20.0, 6.0], dtype=np.float32),
                np.array([-12.0, 10.0, 4.0], dtype=np.float32),
                np.array([15.0, 30.0, 7.0], dtype=np.float32),
            ]

def get_waypoints(type: str) -> list:
    return waypoints[type]
