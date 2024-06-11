try:
    from utils import *
except ImportError:
    from MultiTSP.utils import *

def route_lengths(numNodes, numDrones):
    min_route_length = (numNodes - 1) // numDrones

    route_lens = [min_route_length for _ in range(numDrones)]

    remaining = (numNodes - 1) - sum(route_lens)

    for i in range(remaining):
        route_lens[(i % numDrones)] += 1

    final_route_lens = [x + 1 for x in route_lens]

    return final_route_lens

def fitness(paths: 'list[Path]') -> float:
    path_distances = [path.get_distance() for path in paths]

    total_distance = sum(path_distances)
    max_distance = max(path_distances)

    return total_distance + max_distance

def getNeighbours(paths: 'list[Path]', neighbourhoodSize: int) -> 'list[list[Path]]':
    neighbours: list[list[Path]] = []
    for _ in range(neighbourhoodSize):
        new_paths = create_neighbour(copy.deepcopy(paths))
        neighbours.append(new_paths)

    return neighbours

def create_neighbour(paths: 'list[Path]') -> 'list[Path]':
    new_paths = copy.deepcopy(paths)

    for i in range(len(new_paths)):
        rand_num = random.random()
        if rand_num < 0.4:
            new_paths[i] = two_opt_swap_self(new_paths[i])
        elif rand_num < 0.8:
            other_idx = (random.randint(1, len(new_paths) - 1) + i) % len(new_paths)
            if i == other_idx:
                raise ValueError("Same index selected")
            new_path, other_path = two_opt_swap_other(new_paths[i], new_paths[other_idx])
            new_paths[i] = new_path
            new_paths[other_idx] = other_path
        else:
            continue
    
    return new_paths

def two_opt_swap_self(path: Path) -> Path:
    route_length = len(path.get_path())
    
    a = random.randint(1, route_length - 1)
    b = random.randint(a + 1, route_length)

    path_nodes = path.get_path()

    if random.random() < 0.5:
        new_path = path_nodes[:a] + path_nodes[a:b][::-1] + path_nodes[b:]
    else:
        new_path = path_nodes[:a] + path_nodes[b:] + path_nodes[a:b]

    return Path(new_path)

def two_opt_swap_other(path1: Path, path2: Path) -> 'tuple[Path]':
    route_lengths = [len(path.get_path()) for path in [path1, path2]]
    min_route_length = min(route_lengths)
    
    a_1 = random.randint(1, min_route_length - 1)
    b_1 = random.randint(a_1 + 1, min_route_length)

    diff = b_1 - a_1

    a_2 = random.randint(1, route_lengths[1] - diff)
    b_2 = a_2 + diff

    path1_nodes = path1.get_path()
    path2_nodes = path2.get_path()

    new_path1 = path1_nodes[:a_1] + path2_nodes[a_2:b_2] + path1_nodes[b_1:]
    new_path2 = path2_nodes[:a_2] + path1_nodes[a_1:b_1] + path2_nodes[b_2:]

    return Path(new_path1), Path(new_path2)

def generateInitialSolution(nodes: 'list[Node]', num_drones: int) -> 'list[Path]':
    nodes_wout_start = nodes[1:]
    shuffled_node_idxs = [i for i in range(len(nodes_wout_start))]
    random.shuffle(shuffled_node_idxs)

    paths = [Path([nodes[0]]) for _ in range(num_drones)]

    route_lens = route_lengths(len(nodes), num_drones)

    drone_idx = 0
    for node_idx in shuffled_node_idxs:
        if len(paths[drone_idx].get_path()) < route_lens[drone_idx]:
            paths[drone_idx].add_node(nodes_wout_start[node_idx])
        else:
            drone_idx += 1
            paths[drone_idx].add_node(nodes_wout_start[node_idx])

    return paths


class TabuSearchMultiTSP(AlgoMultiTSP):
    def __init__(self, drones: int, nodes, labels=None):
        super().__init__("tabu", drones, nodes, labels)

    def solve(self, neighbourhoodSize, maxTabuSize, stoppingTurn) -> None:
        random.seed(int.from_bytes(os.urandom(8), 'big'))

        # Generate initial solution
        s0 = generateInitialSolution(self.network.nodes, self.n_drones)

        sBest = s0
        # vBest = fitness(s0)
        self.cost_hist.append(fitness(s0))
        bestCandidate = s0
        tabuList = []
        tabuList.append(s0)

        for _ in tqdm(range(stoppingTurn), desc="Tabu Search Progress"):
            sNeighborhood = getNeighbours(bestCandidate, neighbourhoodSize)
            bestCandidate = sNeighborhood[0]
            for sCandidate in sNeighborhood:
                if (sCandidate not in tabuList) and ((fitness(sCandidate) < fitness(bestCandidate))):
                    bestCandidate = sCandidate

            self.cost_hist.append(fitness(bestCandidate))

            if (fitness(bestCandidate) < fitness(sBest)):
                sBest = bestCandidate
                # vBest = fitness(sBest)
                # best_keep_turn = 0

            tabuList.append(bestCandidate)
            if (len(tabuList) > maxTabuSize):
                tabuList.pop(0)
                    
        self.convert_to_network(sBest)

    def convert_to_network(self, solution:'list[Path]') -> None:
        self.network.paths = solution