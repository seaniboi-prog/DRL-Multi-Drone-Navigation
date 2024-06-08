import numpy as np

cities = dict()

cities["blocks"] = [
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

def get_cities(type: str) -> list:
    return cities[type]
