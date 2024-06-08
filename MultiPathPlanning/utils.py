import pickle
import os
import time
import numpy as np
import matplotlib.pyplot as plt

from constants import PLOT_COLOURS

def save_obj_file(filename: str, obj):
    if os.path.dirname(filename) != '' and not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as outp:
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)
        
def load_obj_file(filename: str):
    with open(filename, 'rb') as inp:
        obj = pickle.load(inp)
    return obj

def thread_test_func(title, fruit, dur):
    print(f"This is {title}")
    time.sleep(5)
    print(f"Processing for {dur} seconds...")
    time.sleep(dur)
    print(f"{title} completed")

    return f"Chosen fruit: {fruit}"

def pop_first_element(arr):
    return arr[0], arr[1:]

def plot_all_routes(targets, drone_paths, filename=None):
    
    plt.figure(figsize=(8, 8))

    for i, path in enumerate(drone_paths):
        route = np.array(path)
        plt.plot(route[:, 0], route[:, 1], c=PLOT_COLOURS[i], zorder=2, label=f"Drone {i+1}")

    np_targets = np.array(targets)
    start, np_targets = pop_first_element(np_targets)
    plt.scatter(start[0], start[1], c='red', marker='x', s=60, zorder=3, label="Start")
    plt.scatter(np_targets[:, 0], np_targets[:, 1], c='black', s=60, marker='x', zorder=3, label="Waypoints")

    plt.legend()
    plt.title("Drone Routes")
    plt.xlabel("X")
    plt.ylabel("Y")
    if filename is not None:
        if os.path.dirname(filename) != '' and not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename)
    plt.show()
