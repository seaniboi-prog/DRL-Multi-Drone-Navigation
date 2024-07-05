import pickle
import dill
import os
import time
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from MultiPathPlanning.constants import PLOT_COLOURS

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

def plot_all_routes(targets, drone_paths, obstacles=[], filename=None):
    
    plt.figure(figsize=(8, 8))
    
    # Plot the obstacles
    for obs in obstacles:
        x_min, y_min, _, x_max, y_max, _ = obs
        plt.gca().add_patch(Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, fill=True, color='grey', alpha=0.8, zorder=1))

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
    
    plt.show(block=False)  # Set block=False to allow code execution to continue
    plt.pause(5)

    # Close the plot window
    plt.close()

def update_multiuav_table(results_table_path, waypoint_type, no_drones, mtsp_algo, rl_algo, action_type, env_variant, results, mtsp_solver):
    total_distance = sum(result["total_distance"] for result in results)
    elapsed_time = max(result["total_time"] for result in results)
    average_time = sum(result["total_time"] for result in results) / len(results)

    # Check if the file exists and read the CSV, otherwise create an empty DataFrame with specified columns
    if os.path.exists(results_table_path):
        results_table = pd.read_csv(results_table_path)
    else:
        results_table = pd.DataFrame(columns=["Slug", "Waypoint Type", "No Drones", "MTSP Algorithm", "RL Algorithm", "Action Type", "Calc Mean Distance",
                                              "Calc Total Distance", "Real Mean Distance", "Real Total Distance", "Average Time", "Elapsed Time"])

    # Construct the slug and row dictionary
    slug = f"{waypoint_type}_{mtsp_algo}_{rl_algo}_{action_type}_{env_variant}_{no_drones}"
    row = {
        "Slug": slug,
        "Waypoint Type": waypoint_type.capitalize(),
        "No Drones": no_drones,
        "MTSP Algorithm": mtsp_algo.upper(),
        "RL Algorithm": rl_algo.upper(),
        "Action Type": action_type.capitalize(),
        "Calc Mean Distance": mtsp_solver.get_total_distance() / no_drones,
        "Calc Total Distance": mtsp_solver.get_total_distance(),
        "Real Mean Distance": total_distance / no_drones,
        "Real Total Distance": total_distance,
        "Average Time": average_time,
        "Elapsed Time": elapsed_time
    }

    # Check if the slug exists in the 'Slug' column and update or append the row
    if slug in results_table["Slug"].values:
        results_table.loc[results_table["Slug"] == slug, :] = pd.DataFrame([row]).values
    else:
        results_table = pd.concat([results_table, pd.DataFrame([row])], ignore_index=True)

    # Save the updated DataFrame back to the CSV file
    results_table.to_csv(results_table_path, index=False)

    return results_table

def display_table(df, title):
    table = Table(title=title)
    rows = df.values.tolist()
    rows = [[str(el) for el in row] for row in rows]
    columns = df.columns.tolist()

    for column in columns:
        table.add_column(column)

    for row in rows:
        table.add_row(*row, style='bright_green')

    console = Console()
    console.print(table)
    
from git import Repo
import os

PATH_OF_GIT_REPO_PC = r'C:\\Users\\seanf\\Documents\\Workspace\\DRL-Multi-Drone-Navigation\\.git'  # make sure .git folder is properly configured
PATH_OF_GIT_REPO_LAPTOP = r'C:\\Users\\seanf\\OneDrive\\Desktop\\School\\DRL-Multi-Drone-Navigation\\.git'  # make sure .git folder is properly configured

def git_pull() -> None:
    try:
        repo = Repo(os.path.abspath(PATH_OF_GIT_REPO_PC))
    except:
        repo = Repo(os.path.abspath(PATH_OF_GIT_REPO_LAPTOP))

    try:
        origin = repo.remote(name='origin')
        origin.pull()
    except:
        print('Some error occurred while pulling the code')

def git_push(commit_message, pull) -> None:
    try:
        repo = Repo(os.path.abspath(PATH_OF_GIT_REPO_PC))
    except:
        repo = Repo(os.path.abspath(PATH_OF_GIT_REPO_LAPTOP))

    try:
        if pull:
            git_pull()

        repo.git.add(all=True)
        repo.index.commit(commit_message)
        origin = repo.remote(name='origin')
        origin.push()
    except:
        print('Some error occurred while pushing the code')