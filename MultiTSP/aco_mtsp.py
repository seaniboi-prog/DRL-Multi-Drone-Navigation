from threading import local
from utils import *

import bisect
import random
import functools
import itertools
from collections import defaultdict

@functools.total_ordering
class Solution:

    def __init__(self, graph, start, ant=None):
        self.graph = graph
        self.start = start
        self.ant = ant
        self.current = start
        self.cost = 0
        self.path = []
        self.nodes = [start]
        self.visited = set(self.nodes)

    def __iter__(self):
        return iter(self.path)

    def __eq__(self, other):
        return self.cost == other.cost

    def __lt__(self, other):
        return self.cost < other.cost

    def __contains__(self, item):
        return item in self.visited or item == self.current

    def __repr__(self):
        easy_id = self.get_easy_id(sep=',', monospace=False)
        return '{}\t{}'.format(self.cost, easy_id)

    def get_easy_id(self, sep=' ', monospace=True):
        nodes = [str(n) for n in self.get_id()]
        if monospace:
            size = max([len(n) for n in nodes])
            nodes = [n.rjust(size) for n in nodes]
        return sep.join(nodes)

    def get_id(self):
        first = min(self.nodes)
        index = self.nodes.index(first)
        return tuple(self.nodes[index:] + self.nodes[:index])

    def add_node(self, node):
        self.nodes.append(node)
        self.visited.add(node)
        self._add_node(node)

    def _add_node(self, node):
        edge = self.current, node
        data = self.graph.edges[edge]
        self.path.append(edge)
        self.cost += data['weight']
        self.current = node

    def close(self):
        self._add_node(self.start)


class Ant:

    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta
        self.drones = None
        self.graph = None
        self.n = None

    def tour(self, graph, drones, start, nodes, opt2):
        self.graph = graph
        self.drones = drones
        self.n = len(graph.nodes)

        solutions = [Solution(graph, start, self) for _ in range(drones)]

        route_lens = [(self.n - 1) // drones for _ in range(drones)]
        for i in range((self.n - 1) % drones):
            route_lens[i] += 1

        unvisited = [node.label for node in nodes]
        for i in range(drones):
            for _ in range(route_lens[i]):
                if len(unvisited) == 0:
                    break
                next_node = self.choose_destination(solutions[i].current, unvisited)
                solutions[i].add_node(next_node)
                unvisited.remove(next_node)
            solutions[i].close()

        if opt2:
            self.opt2_update(graph, opt2, drones, route_lens, solutions)

        return solutions

    def opt2_update(self, graph, opt2, drones, route_lens, solutions):
        for i in range(drones):
            for _ in range(opt2):
                k = route_lens[i] + 1
                while True:
                    a = random.randint(0, k - 1)
                    b = random.randint(0, k - 1)
                    if a != b:
                        break
                if a > b:
                    a, b = b, a
                dist_a = graph.edges[solutions[i].nodes[a], solutions[i].nodes[a + 1]]['weight']
                dist_b = graph.edges[solutions[i].nodes[b], solutions[i].nodes[(b + 1) % k]]['weight']
                dist_c = graph.edges[solutions[i].nodes[a], solutions[i].nodes[b]]['weight']
                dist_d = graph.edges[solutions[i].nodes[a + 1], solutions[i].nodes[(b + 1) % k]]['weight']
                if dist_a + dist_b > dist_c + dist_d:
                    solutions[i].nodes[a + 1:b + 1] = reversed(solutions[i].nodes[a + 1: b + 1])
                    solutions[i].cost += (dist_c + dist_d - dist_a - dist_b)
                    solutions[i].path = []
                    for l in range(k):
                        solutions[i].path.append((solutions[i].nodes[l], solutions[i].nodes[(l + 1) % k]))

    def choose_destination(self, current, unvisited):
        if len(unvisited) == 1:
            return unvisited[0]
        scores = self.get_scores(current, unvisited)
        return self.choose_node(unvisited, scores)

    def choose_node(self, unvisited, scores):
        total = sum(scores)
        cumdist = list(itertools.accumulate(scores))
        index = bisect.bisect(cumdist, random.random() * total)
        return unvisited[min(index, len(unvisited) - 1)]

    def get_scores(self, current, unvisited):
        scores = []
        if self.graph is not None:
            for node in unvisited:
                edge = self.graph.edges[current, node]
                score = self.score_edge(edge)
                scores.append(score)
            return scores

    def score_edge(self, edge):
        weight = edge.get('weight', 1)
        if weight == 0:
            return 1e200
        phe = edge['pheromone']
        return phe ** self.alpha * (1 / weight) ** self.beta
    
class Colony:
    def __init__(self, alpha: float = 1, beta: float = 3):
        self.alpha = alpha
        self.beta = beta

    def get_ants(self, size):
        return [Ant(self.alpha, self.beta) for _ in range(size)]

class ACOMultiTSP(AlgoMultiTSP):
    def __init__(self, n_drones: int, nodes, labels=None):
        super().__init__("ACO", n_drones, nodes, labels)

    def solve(self, alpha: float, beta: float, rho: float, q: float, limit: int, opt2: int, cont: bool = False) -> None:
        random.seed(time.time())

        self.network.init_networkx_graph()

        colony = Colony(alpha, beta)

        ants = colony.get_ants(len(self.network.nodes))

        if not cont:
            self.cost_hist = []
            for u, v in self.network.graph.edges:
                weight = self.network.graph.edges[u, v]['weight']
                if weight == 0:
                    weight = 1e100
                self.network.graph.edges[u, v].setdefault('pheromone', 1 / weight)

        best = None
        best_cost = float('inf')
        
        # pbar = progressbar.ProgressBar()
        for _ in tqdm(range(limit), desc="ACO Progress"):
            # unvisited = [node.label for node in self.network.nodes[1:]]
            drones_solutions = [ant.tour(self.network.graph, self.n_drones, self.network.get_start().label, self.network.nodes[1:], opt2) for ant in ants]
            for solutions in drones_solutions:
                solutions.sort()
            drones_solutions.sort(key=lambda x: sum([y.cost for y in x]))
            self.update_pheromones(drones_solutions, self.network.graph, rho, q)

            local_solution = drones_solutions[0]
            local_dist = sum([s.cost for s in local_solution])
            local_minmax = max([s.cost for s in local_solution])
            local_cost = local_dist + local_minmax
            self.cost_hist.append(local_cost)
            if best is None:
                best = local_solution
                best_cost = sum([s.cost for s in best])
            elif best_cost > local_cost:
                best = local_solution
                best_cost = copy.copy(local_cost)
                        
        if best is not None:
            self.convert_to_network(best)
        else:
            print("No solution found")
    
    def update_pheromones(self, drones_solutions: 'list[list[Solution]]', graph: nx.Graph, rho: float, q: float) -> None:
        next_pheromones = defaultdict(float)
        for solutions in drones_solutions:
            cost = sum([solution.cost for solution in solutions])
            for solution in solutions:
                for path in solution:
                    next_pheromones[path] += q / cost

        for edge in graph.edges:
            curr_pheromone = graph.edges[edge]['pheromone']
            graph.edges[edge]['pheromone'] = curr_pheromone * (1 - rho) + next_pheromones[edge]


    def convert_to_network(self, drone_solutions: 'list[Solution]') -> None:
        self.network.init_paths()
        for i, drone_solution in enumerate(drone_solutions):
            for node in drone_solution.path:
                curr_node: Optional[Node] = self.network.get_node_by_label(node[1])
                if curr_node is not None and not self.network.is_start(curr_node):
                    self.network.add_node_to_path(i, curr_node)
