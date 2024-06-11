try:
    from utils import *
except ImportError:
    from MultiTSP.utils import *

import cvxpy as cp

class CVXPYMultiTSP(AlgoMultiTSP):
    def __init__(self, drones: int, nodes, labels=None):
        super().__init__("cvxpy", drones, nodes, labels)

    def solve(self, verbose=False) -> None:
        random.seed(int.from_bytes(os.urandom(8), 'big'))

        n = len(self.network.nodes)
        dist_matrix = self.build_distance_matrix(n)
    
        # Variables
        X = cp.Variable(dist_matrix.shape, boolean=True)
        u = cp.Variable(n, integer=True)
        ones = np.ones((n, 1))

        # Objective
        objective = cp.Minimize(cp.sum(cp.multiply(dist_matrix, X)))

        # Constraints
        constraints = []
        constraints += [X[0,:] @ ones == self.n_drones]
        constraints += [X[:,0] @ ones == self.n_drones]
        constraints += [X[1:,:] @ ones == 1]
        constraints += [X[:,1:].T @ ones == 1]
        constraints += [cp.diag(X) == 0]
        constraints += [u[1:] >= 2]
        constraints += [u[1:] <= n]
        constraints += [u[0] == 1]

        for i in range(1, n):
            for j in range(1, n):
                if i != j:
                    constraints += [ u[i] - u[j] + 1  <= (n - 1) * (1 - X[i, j]) ]

        # Solution
        problem = cp.Problem(objective, constraints)
        problem.solve(verbose=verbose)

        # Get solution
        X_sol = np.argwhere(X.value == 1)

        # Create paths
        self.convert_to_network(X_sol)

    def convert_to_network_old(self, solution: np.ndarray) -> None:
        for i in range(self.n_drones):
            j = i
            a = 10e10
            while a != 0:
                a = solution[j, 1]
                if a != 0:
                    self.network.add_node_to_path(i, self.network.nodes[a])
                j = np.where(solution[:, 0] == a)[0][0]
                a = j
                
    def convert_to_network(self, solution: np.ndarray) -> None:
        min_nodes_per_drone = ((len(self.network.nodes) - 1) // self.n_drones)
        max_nodes_per_drone = ((len(self.network.nodes) - 1) // self.n_drones) + 1

        drone_idx = 0
        X_counter = 0
        j = drone_idx
        a = 10e10
        path_length = 0
        nodes_added = 0
        
        while a != 0:
            a = solution[j, 1]
            if a != 0:
                self.network.add_node_to_path(drone_idx, self.network.nodes[a])
                path_length += 1
                nodes_added += 1
            j = np.where(solution[:, 0] == a)[0][0]
            a = j
            
            if path_length >= max_nodes_per_drone:
                drone_idx += 1
                path_length = 0
                
            if a == 0:
                X_counter += 1
                j = X_counter
                a = 10e10
                if path_length >= min_nodes_per_drone:
                    drone_idx += 1
                    path_length = 0
                    
            if drone_idx >= self.n_drones or nodes_added == len(self.network.nodes) - 1:
                break