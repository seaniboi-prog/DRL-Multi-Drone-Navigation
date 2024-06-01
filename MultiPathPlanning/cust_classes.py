from utils import *
import numpy as np

class DronePaths:
    def __init__(self, restore_path=None) -> None:
        self.paths: list = []
        if restore_path is not None:
            self.restore_paths(restore_path)

    def add_path(self, path: 'list[np.ndarray]') -> None:
        self.paths.append(path)

    def get_path(self, drone_id: int) -> list:
        return self.paths[drone_id]
    
    def get_paths(self) -> 'list[np.ndarray]':
        return self.paths

    def save_paths(self, path: str) -> None:
        save_obj_file(path, self)

    def restore_paths(self, path: str) -> None:
        loaded_obj = load_obj_file(path)
        self.__dict__.update(loaded_obj.__dict__)

    def __str__(self) -> str:
        output = ""
        for drone_id, path in enumerate(self.paths):
            output += f"Drone {(drone_id+1)}: "
            for point in path:
                output += f"({point[0]}, {point[1]}, {point[2]}) -> "
            output = output[:-3] + "\n"
        return output
    
if __name__ == "__main__":
    path = DronePaths()
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    c = np.array([7, 8, 9])
    d = np.array([10, 11, 12])
    e = np.array([13, 14, 15])
    f = np.array([16, 17, 18])
    path.add_path([a, e, c, a])
    path.add_path([a, d, b, f, a])
    path.save_paths('test.pkl')

    restored_path = DronePaths('test.pkl')
    print(restored_path)