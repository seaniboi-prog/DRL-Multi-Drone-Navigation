from utils import *

class DRLMultiTSP(AlgoMultiTSP):
    def __init__(self, drones: int, nodes, labels=None):
        super().__init__("DRL", drones, nodes, labels)

    def solve(self) -> None:
        random.seed(time.time())

    def convert_to_network(self, solution: np.ndarray) -> None:
        pass