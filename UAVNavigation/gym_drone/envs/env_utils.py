import numpy as np
import random

from typing import Union

def unit_vector(vector: np.ndarray):
    return vector / np.linalg.norm(vector)
    
def normalize_arr(arr: np.ndarray, method: str = "minmax") -> np.ndarray:
    if np.all(arr == 0):
        return arr
    
    if method == "minmax":
        min_val = np.min(arr)
        max_val = np.max(arr)
        return (arr - min_val) / (max_val - min_val)
    elif method == "mean":
        mean_val = np.mean(arr)
        return (arr - mean_val) / mean_val
    elif method == "zscore":
        mean_val = np.mean(arr)
        std_dev = np.std(arr)
        return (arr - mean_val) / std_dev
    else:
        raise ValueError("Invalid normalization method: {}".format(method))
    

def cust_rand_coor(low: Union[float, list], high: Union[float, list], corner: int, z_range = [5., 15.]):
    
    if corner == 0:
        x_sign = 1
        y_sign = 1
    elif corner == 1:
        x_sign = -1
        y_sign = 1
    elif corner == 2:
        x_sign = -1
        y_sign = -1
    else:
        x_sign = 1
        y_sign = -1

    if isinstance(low, list) and isinstance(high, list):
        x_low, y_low = low
        x_high, y_high = high

        x_value = random.uniform(x_low, x_high) * x_sign
        y_value = random.uniform(y_low, y_high) * y_sign

    elif isinstance(low, float) and isinstance(high, float):
        x_value = random.uniform(low, high) * x_sign
        y_value = random.uniform(low, high) * y_sign

    else:
        raise ValueError("Invalid low and high values: {}, {}".format(low, high))
    
    z_value = random.uniform(z_range[0], z_range[1])

    return x_value, y_value, z_value

def blocks_rand_coor(corner: int):
    if corner == 0:
        x_value = random.uniform(20, 65)
        y_value = random.uniform(21, 45)
        z_value = random.uniform(10, 20)
    elif corner == 1:
        x_value = random.uniform(65, 80)
        y_value = random.uniform(-25, 20)
        z_value = random.uniform(5, 15)
    elif corner == 2:
        x_value = random.uniform(20, 65)
        y_value = random.uniform(-45, -25)
        z_value = random.uniform(5, 15)
    elif corner == 3:
        x_value = random.uniform(-15, 15)
        y_value = random.uniform(-45, 45)
        z_value = random.uniform(5, 15)
    
    return x_value, y_value, z_value

def get_random_coors(low: Union[float, list], high: Union[float, list], count: int, z_range = [5., 15.], num_coors = 1, env_variant = "cust") -> list:
    random.seed(count)
    
    list_coors = []
    
    corner = count % 4
    
    for _ in range(num_coors):
        if env_variant == "cust":
            corner = (corner + 1) % 4
            x_value, y_value, z_value = cust_rand_coor(low, high, corner, z_range)
        elif env_variant == "blocks":
            x_value, y_value, z_value = blocks_rand_coor(corner)

        list_coors.append(np.array([x_value, y_value, z_value]))
    
    return list_coors