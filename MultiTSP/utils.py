import os
import pickle
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
import networkx as nx
from abc import ABC, abstractmethod
from typing import Union

from rich.console import Console
from rich.table import Table

# Global imports
import random
import time
import copy
from typing import Optional
import progressbar
from tqdm import tqdm

COLOURS = ['r', 'g', 'b', 'gold', 'orange', 'c', 'm', 'y']

def euc_distance(point1: np.ndarray, point2: np.ndarray) -> float:
    return np.sqrt(np.sum((point1 - point2)**2))

class Node:
    def __init__(self, x, y, z, l):
        self.x = x
        self.y = y
        self.z = z
        self.label = l

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, Node):
            return False
        return self.x == __value.x and self.y == __value.y and self.z == __value.z and self.label == __value.label

    def get_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])
    
    def __str__(self) -> str:
        return f"({round(self.x,2)}, {round(self.y,2)}, {round(self.z,2)})"
    
    def dist_from(self, other: 'Node') -> float:
        return euc_distance(self.get_array(), other.get_array())
    
class Path:
    def __init__(self, route=[]):
        self.route: list[Node] = route

    def get_start(self) -> Union[Node, None]:
        if len(self.route) == 0:
            return None
        else:
            return self.route[0]
        
    def set_path(self, path) -> None:
        self.route = path

    def get_path(self) -> 'list[Node]':
        return self.route

    def add_node(self, node: Node) -> None:
        self.route.append(node)

    def get_distance(self) -> float:
        length = 0
        for i in range(len(self.route)):
            if i == len(self.route) - 1:
                length += self.route[i].dist_from(self.route[0])
            else:
                length += self.route[i].dist_from(self.route[i+1])
        return length
    
    def get_time(self, speed: float) -> float:
        return self.get_distance() / speed
    
    def __str__(self) -> str:
        return " -> ".join([node.label for node in self.route])

class Network:
    def __init__(self, drones: int, nodes, labels=None):
        self.n_drones = drones
        self.nodes = []
        self.graph = nx.Graph()
        self.generate_nodes(nodes, labels)

        self.start: Node = self.nodes[0]
        self.paths: list[Path] = [Path([self.start]) for _ in range(drones)]

        self.init_networkx_graph()

    def generate_nodes(self, nodes, labels) -> None:
        if labels is not None:
            self.nodes = [Node(coor[0], coor[1], coor[2], label) for coor, label in zip(nodes, labels)]
        else:
            self.nodes = [Node(coor[0], coor[1], coor[2], i) for i, coor in enumerate(nodes)]

    def init_networkx_graph(self) -> None:
        self.graph = nx.Graph()

        self.graph.add_nodes_from([(node.label, {'pos': (node.x, node.y)}) for node in self.nodes])
        
        for i in range(len(self.nodes)):
            for j in range(i+1, len(self.nodes)):
                self.graph.add_edge(self.nodes[i].label, self.nodes[j].label,
                                    weight=self.nodes[i].dist_from(self.nodes[j]))

    def init_paths(self) -> None:
        self.paths: list[Path] = [Path([self.start]) for _ in range(self.n_drones)]

    def get_start(self) -> Node:
        return self.start
    
    def get_graph(self) -> nx.Graph:
        return self.graph
    
    def is_start(self, node: Node) -> bool:
        return node == self.get_start()

    def get_node_by_label(self, label) -> Union[Node, None]:
        for node in self.nodes:
            if node.label == label:
                return node

        return None

    def get_paths(self) -> 'list[Path]':
        return self.paths
    
    def get_paths_list(self, includeStart=True) -> 'list[list[np.ndarray]]':
        paths_list = []
        for path in self.paths:
            if includeStart:
                paths_list.append([node.get_array() for node in path.get_path()])
            else:
                paths_list.append([node.get_array() for node in path.get_path()[1:]])
        return paths_list

    def set_path(self, idx: int, path: Path) -> None:
        self.paths[idx].set_path(path)

    def print_paths(self) -> None:
        for i, path in enumerate(self.paths):
            print(f"Drone {i+1}: {path}")

    def add_node_to_path(self, idx: int, node: Node) -> None:
        self.paths[idx].add_node(node)

    def get_total_dist(self) -> float:
        return sum([path.get_distance() for path in self.paths])
    
    def get_all_dists(self) -> 'list[float]':
        return [path.get_distance() for path in self.paths]
    
    def get_minmax_dist(self) -> float:
        return max([path.get_distance() for path in self.paths])
    
    def get_score(self) -> float:
        return self.get_total_dist() + self.get_minmax_dist()

    def get_total_time(self, speed: float) -> float:
        drone_dists = [path.get_distance() for path in self.paths]
        max_drone_dist = max(drone_dists)

        return max_drone_dist / speed
    
    def display_graph(self, title: str = "Graph", edges=False, positions="spring") -> None:
        plt.figure(figsize=(12, 12))
        if positions == "spring":
            pos = nx.spring_layout(self.graph, seed=42)
        else:
            pos = nx.get_node_attributes(self.graph, 'pos')
        nx.draw(self.graph, pos, with_labels=True, node_size=700, node_color='lightblue')
        if edges:
            labels = nx.get_edge_attributes(self.graph, 'weight')
            nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=labels, font_size=7)
        plt.title(title)
        # Display the plot for a few seconds
        plt.show(block=False)  # Set block=False to allow code execution to continue

        # Pause for a few seconds (e.g., 3 seconds)
        plt.pause(3)

        # Close the plot window
        plt.close()
        
    
    def plot_network(self, title: str = "Network") -> None:
        plt.figure(figsize=(8, 8))
        
        plt.scatter(self.get_start().x, self.get_start().y, c='brown', s=600, zorder=2, marker=MarkerStyle(marker='*'))

        n_x = [node.x for node in self.nodes if not self.is_start(node)]
        n_y = [node.y for node in self.nodes if not self.is_start(node)]
        plt.scatter(n_x, n_y, c='k', s=500, zorder=2)

        l_n = len(self.nodes)
        
        x_text_h = { 1: {
                        20: 3,
                        50: 8,
                        100: 16,
                        200: 45
                    },
                     2: {
                        20: 3,
                        50: 15,
                        100: 20,
                        200: 65
                     }
                }
        
        y_text_v = { 1: {
                        20: 4,
                        50: 12,
                        100: 16,
                        200: 45
                    },
                     2: {
                        20: 4,
                        50: 12,
                        100: 16,
                        200: 45
                     }
                }
        
        for node in self.nodes:
            l_l = len(str(node.label))
            plt.text(node.x - x_text_h[l_l][l_n], node.y - y_text_v[l_l][l_n], str(node.label), color='w', fontsize=12, zorder=3)
        
        plt.title(title)
        # Display the plot for a few seconds
        plt.show(block=False)  # Set block=False to allow code execution to continue

        plt.pause(3)
        plt.close()
        

    def plot_paths(self, title: str = "Paths", pause: int = 3, filename = None) -> None:
        plt.figure(figsize=(8, 8))
        
        plt.scatter(self.get_start().x, self.get_start().y, c='red', s=600, zorder=2, marker=MarkerStyle(marker='*'))

        n_x = [node.x for node in self.nodes if not self.is_start(node)]
        n_y = [node.y for node in self.nodes if not self.is_start(node)]
        plt.scatter(n_x, n_y, c='k', s=500, zorder=2)
        
        l_n = len(self.nodes)
        
        x_text_h = { 1: {
                        20: 3,
                        50: 8,
                        100: 16,
                        200: 45
                    },
                     2: {
                        20: 3,
                        50: 15,
                        100: 20,
                        200: 65
                     }
                }
        
        y_text_v = { 1: {
                        20: 4,
                        50: 12,
                        100: 16,
                        200: 45
                    },
                     2: {
                        20: 4,
                        50: 12,
                        100: 16,
                        200: 45
                     }
                }
        
        for node in self.nodes:
            l_l = len(str(node.label))
            plt.text(node.x - x_text_h[l_l][l_n], node.y - y_text_v[l_l][l_n], str(node.label), color='w', fontsize=12, zorder=3)
        
        for i, path in enumerate(self.paths):
            x = [node.x for node in path.get_path()]
            y = [node.y for node in path.get_path()]
            x.append(self.get_start().x)
            y.append(self.get_start().y)
            plt.plot(x, y, COLOURS[i], zorder=1, label=f"Drone {i+1}")
        
        plt.legend(loc='best')
        plt.title(title)
        
        if filename is not None:
            if os.path.dirname(filename) != '' and not os.path.exists(os.path.dirname(filename)):
                os.makedirs(os.path.dirname(filename), exist_ok=True)
            plt.savefig(filename)

        # Pause for a few seconds (e.g., 3 seconds)
        if pause > 0:
            plt.show(block=False)  # Set block=False to allow code execution to continue
            plt.pause(pause)

            # Close the plot window
            plt.close()
        else:
            plt.show(block=True)
        
    def plot_sub_paths(self, axs, index: tuple, title: str) -> None:
        axs[index].scatter(self.get_start().x, self.get_start().y, c='red', s=500, zorder=2, marker='*')

        n_x = [node.x for node in self.nodes if not self.is_start(node)]
        n_y = [node.y for node in self.nodes if not self.is_start(node)]
        axs[index].scatter(n_x, n_y, c='k', s=400, zorder=2)
        
        for i, node in enumerate(self.nodes):
            axs[index].text(node.x - 0.7, node.y - 1, str(node.label), color='w', fontsize=12, zorder=3)

        for i, path in enumerate(self.paths):
            x = [node.x for node in path.get_path()]
            y = [node.y for node in path.get_path()]
            x.append(self.get_start().x)
            y.append(self.get_start().y)
            axs[index].plot(x, y, COLOURS[i], zorder=1, label=f"Drone {i+1}")
        
        axs[index].legend(loc='best')
        axs[index].set_title(title)

class AlgoMultiTSP(ABC):
    def __init__(self, algo, n_drones: int, nodes, labels=None):
        self.algorithm: str = algo
        self.n_drones: int = n_drones
        self.network = Network(n_drones, nodes, labels)
        self.cost_hist = []

    @abstractmethod
    def solve(self, *args, **kwargs):
        pass

    @abstractmethod
    def convert_to_network(self, *args, **kwargs):
        pass
    
    def build_distance_matrix(self, n) -> np.ndarray:
        dist_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                dist_matrix[i, j] = self.network.nodes[i].dist_from(self.network.nodes[j])
        
        return dist_matrix

    def plot_progress(self, pause: int = 3) -> None:
        plt.figure(figsize=(12, 6))
        xaxis = np.arange(len(self.cost_hist))
        plt.plot(xaxis, self.cost_hist, 'b-')
        plt.title(f"{self.algorithm.upper()}: Cost Progress")
        # Display the plot for a few seconds
        plt.show(block=False)  # Set block=False to allow code execution to continue

        # Pause for a few seconds (e.g., 3 seconds)
        plt.pause(pause)

        # Close the plot window
        plt.close()
    
    def draw_network(self) -> None:
        self.network.display_graph(f"{self.algorithm.upper()}: Weight Graph")

    def plot_solution(self, pause: int = 3, filename=None) -> None:
        self.network.plot_paths(f"{self.algorithm.upper()}: Paths", pause, filename)
        print(f"Total score: {round(self.network.get_score(), 2)}")
        print(f"Longest salesman distance: {round(self.network.get_minmax_dist(), 2)}")
        print(f"Total distance: {round(self.network.get_total_dist(), 2)}")
        
    def plot_sub_solution(self, axs, index: tuple, title: str) -> None:
        self.network.plot_sub_paths(axs, index, title)
        
    def print_paths(self) -> None:
        self.network.print_paths()

    def get_total_distance(self) -> float:
        return self.network.get_total_dist()
    
    def get_all_dists(self) -> 'list[float]':
        return self.network.get_all_dists()
    
    def get_minmax_distance(self) -> float:
        return self.network.get_minmax_dist()
    
    def get_score(self) -> float:
        return self.network.get_score()
    
    # TODO: Calculate Distance Squared
    def get_dist_square(self) -> float:
        pass
    
    def get_paths(self) -> 'list[Path]':
        return self.network.get_paths()
    
    def get_paths_list(self, includeStart=True) -> 'list[list[np.ndarray]]':
        return self.network.get_paths_list(includeStart)
    

def compare_solution_scores(mtsp_solver: AlgoMultiTSP, city_type) -> Union[str,None]:
    no_drones = mtsp_solver.n_drones
    filename = os.path.join("mtsp_results", "paths", city_type, f"{no_drones}_drones", f"{city_type}_{no_drones}_{mtsp_solver.algorithm}_best_solution.pkl")
    plot_path = os.path.join("mtsp_results", "plots", city_type, f"{no_drones}_drones", f"{city_type}_{no_drones}_{mtsp_solver.algorithm}_solution.png")
    
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            prev_best_solver: AlgoMultiTSP = pickle.load(f)
        
        if mtsp_solver.get_score() < prev_best_solver.get_score():
            with open(filename, 'wb') as f:
                pickle.dump(mtsp_solver, f)
            return plot_path
        else:
            return None
                
    else:
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'wb') as f:
            pickle.dump(mtsp_solver, f)
        return plot_path

def update_mtsp_table(results_table_path, city_type, no_drones, mtsp_algo, mtsp_solver: AlgoMultiTSP, time) -> pd.DataFrame:
    # Check if the file exists and read the CSV, otherwise create an empty DataFrame with specified columns
    if os.path.exists(results_table_path):
        results_table = pd.read_csv(results_table_path)
    else:
        results_table = pd.DataFrame(columns=["Slug", "Waypoint Type", "No. Drones", "No. Targets", "MTSP Algorithm", "Min Distance", "Mean Distance", "Max Distance", "Total Distance", "Score", "Time"])

    num_nodes = len(mtsp_solver.network.nodes)

    # Construct the slug and row dictionary
    slug = f"{city_type}_{num_nodes}_{mtsp_algo}_{no_drones}"
    row = {
        "Slug": slug,
        "Waypoint Type": city_type.capitalize(),
        "No. Drones": no_drones,
        "No. Targets": num_nodes,
        "MTSP Algorithm": mtsp_algo.upper(),
        "Min Distance": min(mtsp_solver.get_all_dists()),
        "Mean Distance": mtsp_solver.get_total_distance() / no_drones,
        "Max Distance": max(mtsp_solver.get_all_dists()),
        "Total Distance": mtsp_solver.get_total_distance(),
        "Score": mtsp_solver.get_score(),
        "Time": time
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
    
def generate_labels(num_labels):
    labels = []
    letters = string.ascii_uppercase
    while num_labels > 0:
        num_labels -= 1
        labels.append(letters[num_labels % 26])
        num_labels //= 26

    return ''.join(reversed(labels))

def get_labels(num_nodes):
    labels = []
    for i in range(num_nodes):
        labels.append(generate_labels(i + 1))
    return labels