from utils import *

class Chromosome():
    
    # Random generated Chromosome
    #  m - number of traveling salesmans
    def __init__(self, number_of_cities, number_of_traveling_salesman, adj):
        self.n = number_of_cities
        self.m = number_of_traveling_salesman
        self.adj = adj
        c = np.array(range(1,number_of_cities))
        np.random.shuffle(c)
        self.solution = np.array_split(c, self.m)
        for i in range(len(self.solution)):
            self.solution[i] = np.insert(self.solution[i],0,0)
            self.solution[i] = np.append(self.solution[i],0)
        self.fitness()
            
    # Evaluate the Chromosome - Fitness function
    #  based on 2 features: 
    #   - overall cost (cumulated from all salesman)
    #   - worst (longest) salesman cost
    #  adj - adjacency matrix
    def fitness(self):
        self.cost = 0
        longest_salesman_fitness = []
        longest_salesman_length = 0
        for i in range(self.m):
            salesman = self.solution[i]
            salesman_fitness = 0
            for j in range(len(salesman) - 1):
                salesman_fitness = salesman_fitness + self.adj[salesman[j]][salesman[j+1]]
            self.cost = self.cost + salesman_fitness
            if len(salesman) > longest_salesman_length or (len(salesman) == longest_salesman_length and salesman_fitness > self.minmax):
                longest_salesman_length = len(salesman)
                self.minmax = salesman_fitness
        self.score = self.cost + self.minmax
    
    # Mutation operator - mutates a single Traveling Salesman
    #  by swaping 2 cities
    def mutate_local(self):
        index = np.random.randint(0,self.m)
        mutant = self.solution[index]
        i,j = np.random.randint(1,len(mutant)-1), np.random.randint(1,len(mutant)-1)
        mutant[i], mutant[j] = mutant[j], mutant[i]
        old_cost = self.cost
        self.fitness()
    
    # Mutation operator - mutates 2 Traveling Salesmans
    #  by removing a city from a salesman and asigning it to the second one
    def mutate_global(self):
        for i in range(self.m):
            if len(self.solution[i]) < 3:
                print(i, self.solution[i])
        
        
        index1, index2 = np.random.randint(0,self.m), np.random.randint(0,self.m)
        while index1 == index2:
            index1, index2 = np.random.randint(0,self.m), np.random.randint(0,self.m)
        while len(self.solution[index1]) < 4:
            index1, index2 = np.random.randint(0,self.m), np.random.randint(0,self.m)
        mutant1, mutant2 = self.solution[index1], self.solution[index2]
        i,j = np.random.randint(1,len(mutant1)-1), np.random.randint(1,len(mutant2)-1)
        self.solution[index2] = np.insert(mutant2, j, mutant1[i])
        self.solution[index1] = np.delete(mutant1, i)
        old_cost = self.cost
        self.fitness()

class HillClimbMultiTSP(AlgoMultiTSP):
    def __init__(self, drones: int, nodes, labels=None):
        super().__init__("Hill Climbing", drones, nodes, labels)

    def solve(self, epochs, cont=False) -> None:
        n = len(self.network.nodes)
        
        if cont:
            chromosome = copy.deepcopy(self.best_chromosome)
        else:
            chromosome = Chromosome(number_of_cities = n, number_of_traveling_salesman = self.n_drones, adj = self.build_distance_matrix(n))
        
        for _ in tqdm(range(epochs), desc="Hill Climbing Progress"):
            # Mutate globally
            chromosome_copy = copy.deepcopy(chromosome)
            chromosome_copy.mutate_global()
            if chromosome_copy.score < chromosome.score:
                chromosome = chromosome_copy
            # Mutate locally
            chromosome_copy = copy.deepcopy(chromosome)
            chromosome_copy.mutate_local()
            if chromosome_copy.score < chromosome.score:
                chromosome = chromosome_copy
                
            self.cost_hist.append(chromosome.score)
            
        self.best_chromosome = chromosome
        
        print("Solution")
        print(chromosome.solution)
        self.convert_to_network(chromosome)

    def convert_to_network(self, chromosome: Chromosome) -> None:
        self.network.init_paths()
        
        for i, drone_sol in enumerate(chromosome.solution):
            for j in range(len(drone_sol)):
                if drone_sol[j] != 0:
                    self.network.add_node_to_path(i, self.network.nodes[drone_sol[j]])