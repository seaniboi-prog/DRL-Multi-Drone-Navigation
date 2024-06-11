try:
    from utils import *
except ImportError:
    from MultiTSP.utils import *

def random_range(n, total):
    """Return a randomly chosen list of n positive integers summing to total.
    Each such list is equally likely to occur."""

    dividers = sorted(random.sample(range(1, total), n - 1))
    return [a - b for a, b in zip(dividers + [total], [0] + dividers)]

# Randomly distribute number of dustbins to subroutes
# Maximum and minimum values are maintained to reach optimal result
def route_lengths(numNodes, numDrones):
    upper = (numNodes + numDrones - 1)
    fa = upper/numDrones*1.6 # max route length
    fb = upper/numDrones*0.6 # min route length
    a = random_range(numDrones, upper)
    while 1:
        if all( i < fa and i > fb  for i in a):
                break
        else:
                a = random_range(numDrones, upper)
    return a

class Dustbin:
    # Good old constructor
    def __init__ (self, x = None, y = None, z = None, label=None, isNull = False):
        self.x = x
        self.y = y
        self.z = z
        self.label = label
        self.isNull = isNull

    def getX (self):
        return self.x

    def getY (self):
        return self.y
    
    def getZ (self):
        return self.z
    
    def getLabel (self):
        return self.label
    
    def set_coors(self, x, y, z):
        self.isNull = False
        self.x = x
        self.y = y
        self.z = z

    def get_coors(self):
        return np.array([self.x, self.y, self.z])

	# Returns distance to the dustbin passed as argument
    def distanceTo (self, db: 'Dustbin'):
        return euc_distance(self.get_coors(), db.get_coors())

	# Gives string representation of the Object with coordinates
    def toString (self):
        s =  '(' + str(self.getX()) + ',' + str(self.getY()) + ')'
        return s

	# Check if cordinates have been assigned or not
	# Dusbins with (-1, -1) as coordinates are created during creation on chromosome objects
    def checkNull(self):
        return self.isNull
    
class RouteManager:
    destinationDustbins = []

    @classmethod
    def addDustbin (cls, db):
        cls.destinationDustbins.append(db)

    @classmethod
    def getDustbin (cls, index):
        return cls.destinationDustbins[index]

    @classmethod
    def numberOfDustbins(cls):
        return len(cls.destinationDustbins)
    
class Route:
    # Good old constructor
    def __init__ (self, numNodes, numDrones, route = None):
        # 2D array which is collection of respective routes taken by trucks
        self.route = []
        # 1D array having routes in a series - used during crossover operation
        self.base = []
        # 1D array having route lengths
        self.routeLengths = route_lengths(numNodes, numDrones)

        self.numDrones = numDrones
        self.numNodes = numNodes

        for _ in range(numDrones):
            self.route.append([])

        # fitness value and total distance of all routes
        # self.fitness = 0
        # self.distance = 0

        # creating empty route
        if route == None:
            for i in range(RouteManager.numberOfDustbins()-1):
                self.base.append(Dustbin(isNull=True))

        else:
            self.route = route

    def generateIndividual (self):
        k=0
        # put 1st member of RouteManager as it is (It represents the initial node) and shuffle the rest before adding
        for dindex in range(1, RouteManager.numberOfDustbins()):
            self.base[dindex-1] = RouteManager.getDustbin(dindex)
        random.shuffle(self.base)

        for i in range(self.numDrones):
            self.route[i].append(RouteManager.getDustbin(0)) # add same first node for each route
            for _ in range(self.routeLengths[i]-1):
                self.route[i].append(self.base[k]) # add shuffled values for rest
                k+=1

    # Returns j'th dustbin in i'th route
    def getDustbin(self, i, j):
        return self.route[i][j]

    # Sets value of j'th dustbin in i'th route
    def setDustbin(self, i, j, db):
        self.route[i][j] = db
        #self.route.insert(index, db)
        self.fitness = 0
        self.distance = 0

    # Returns the fitness value of route
    def getFitness(self):
        fitness = 1/self.getScore()

        return fitness

    # Return total ditance covered in all subroutes
    def getDistance(self):
        routeDistance = 0

        for i in range(self.numDrones):
            for j in range(self.routeLengths[i]):
                fromDustbin = self.getDustbin(i, j)

                if j+1 < self.routeLengths[i]:
                    destinationDustbin = self.getDustbin(i, j + 1)

                else:
                    destinationDustbin = self.getDustbin(i, 0)

                routeDistance += fromDustbin.distanceTo(destinationDustbin)

        return routeDistance
    
    def getMinMax(self):
        minmax = 0
        for i in range(self.numDrones):
            routeDistance = 0
            for j in range(self.routeLengths[i]):
                fromDustbin = self.getDustbin(i, j)
                if j+1 < self.routeLengths[i]:
                    destinationDustbin = self.getDustbin(i, j + 1)
                else:
                    destinationDustbin = self.getDustbin(i, 0)
                routeDistance += fromDustbin.distanceTo(destinationDustbin)
            if routeDistance > minmax:
                minmax = routeDistance
        return minmax
    
    def getScore(self):
        return self.getDistance() + self.getMinMax()

    # Checks if the route contains a particular dustbin
    def containsDustbin(self, db):
        if db in self.base: #base <-> route
            return True
        else:
            return False

    # Returns route in the form of a string
    def toString (self):
        geneString = '|'
        print (self.routeLengths)
        #for k in range(RouteManager.numberOfDustbins()-1):
        #    print (self.base[k].toString())
        for i in range(self.numDrones):
            for j in range(self.routeLengths[i]):
                geneString += self.getDustbin(i,j).toString() + '|'
            geneString += '\n'

        return geneString
    
class Population:
    routes = []
    # Good old contructor
    def __init__ (self, populationSize, initialise, numNodes, numDrones):
        self.populationSize = populationSize
        if initialise:
            for _ in range(populationSize):
                newRoute = Route(numNodes, numDrones) # Create empty route
                newRoute.generateIndividual() # Add route sequences
                self.routes.append(newRoute) # Add route to the population

    # Saves the route passed as argument at index
    def saveRoute (self, index, route):
        self.routes[index] = route

    # Returns route at index
    def getRoute (self, index):
        return self.routes[index]

    # Returns route with maximum fitness value
    def getFittest (self):
        fittest = self.routes[0]

        for i in range(1, self.populationSize):
            if fittest.getFitness() <= self.getRoute(i).getFitness():
                fittest = self.getRoute(i)

        return fittest

    # Equate current population values to that of pop
    def equals(self, pop):
        self.routes = pop.routes

class GA:

    @classmethod
    # Evolve pop
    def evolvePopulation(cls, pop, numNodes, numDrones, elitism, mutationRate, tournamentSize):

        newPopulation = Population(pop.populationSize, False, numNodes, numDrones)

        elitismOffset = 0
        # If fittest chromosome has to be passed directly to next generation
        if elitism:
            newPopulation.saveRoute(0, pop.getFittest())
            elitismOffset = 1

        # Performs tournament selection followed by crossover to generate child
        for i in range(elitismOffset, newPopulation.populationSize):
            parent1 = cls.tournamentSelection(pop, numNodes, numDrones, tournamentSize)
            parent2 = cls.tournamentSelection(pop, numNodes, numDrones, tournamentSize)
            child = cls.crossover(parent1, parent2, numNodes, numDrones)
            # Adds child to next generation
            newPopulation.saveRoute(i, child)


        # Performs Mutation
        for i in range(elitismOffset, newPopulation.populationSize):
            cls.mutate(newPopulation.getRoute(i), numDrones, mutationRate)

        return newPopulation

    # Function to implement crossover operation
    @classmethod
    def crossover (cls, parent1, parent2, numNodes, numDrones):
        child = Route(numNodes, numDrones)
        child.base.append(Dustbin(isNull=True)) # since size is (numNodes - 1) by default
        startPos = 0
        endPos = 0
        while (startPos >= endPos):
            startPos = random.randint(1, numNodes-1)
            endPos = random.randint(1, numNodes-1)

        parent1.base = [parent1.route[0][0]]
        parent2.base = [parent2.route[0][0]]

        for i in range(numDrones):
            for j in range(1, parent1.routeLengths[i]):
                parent1.base.append(parent1.route[i][j])


        for i in range(numDrones):
            for j in range(1, parent2.routeLengths[i]):
                parent2.base.append(parent2.route[i][j])

        for i in range(1, numNodes):
            if i > startPos and i < endPos:
                child.base[i] = parent1.base[i]

        for i in range(numNodes):
            if not(child.containsDustbin(parent2.base[i])):
                for i1 in range(numNodes):
                    if child.base[i1].checkNull():
                        child.base[i1] =  parent2.base[i]
                        break

        k=0
        child.base.pop(0)
        for i in range(numDrones):
            child.route[i].append(RouteManager.getDustbin(0)) # add same first node for each route
            for j in range(child.routeLengths[i]-1):
                child.route[i].append(child.base[k]) # add shuffled values for rest
                k+=1
        return child

    # Mutation opeeration
    @classmethod
    def mutate (cls, route, numDrones, mutationRate):
        index1 = 0
        index2 = 0
        while index1 == index2:
            index1 = random.randint(0, numDrones - 1)
            index2 = random.randint(0, numDrones - 1)
        #print ('Indexes selected: ' + str(index1) + ',' + str(index2))

        #generate replacement range for 1
        route1startPos = 0
        route1lastPos = 0
        while route1startPos >= route1lastPos or route1startPos == 1:
            route1startPos = random.randint(1, route.routeLengths[index1] - 1)
            route1lastPos = random.randint(1, route.routeLengths[index1] - 1)

        #generate replacement range for 2
        route2startPos = 0
        route2lastPos = 0
        while route2startPos >= route2lastPos or route2startPos == 1:
            route2startPos = random.randint(1, route.routeLengths[index2] - 1)
            route2lastPos= random.randint(1, route.routeLengths[index2] - 1)

        #print ('startPos, lastPos: ' + str(route1startPos) + ',' + str(route1lastPos) + ',' + str(route2startPos) + ',' + str(route2lastPos))
        swap1 = [] # values from 1
        swap2 = [] # values from 2

        if random.randrange(1) < mutationRate:
            # pop all the values to be replaced
            for i in range(route1startPos, route1lastPos + 1):
                swap1.append(route.route[index1].pop(route1startPos))

            for i in range(route2startPos, route2lastPos + 1):
                swap2.append(route.route[index2].pop(route2startPos))

            del1 = (route1lastPos - route1startPos + 1)
            del2 = (route2lastPos - route2startPos + 1)

            # add to new location by pushing
            route.route[index1][route1startPos:route1startPos] = swap2
            route.route[index2][route2startPos:route2startPos] = swap1

            route.routeLengths[index1] = len(route.route[index1])
            route.routeLengths[index2] = len(route.route[index2])

    # Tournament Selection: choose a random set of chromosomes and find the fittest among them 
    @classmethod
    def tournamentSelection (cls, pop, numNodes, numDrones, tournamentSize):
        tournament = Population(tournamentSize, False, numNodes, numDrones)

        for i in range(tournamentSize):
            randomInt = random.randint(0, pop.populationSize-1)
            tournament.saveRoute(i, pop.getRoute(randomInt))

        fittest = tournament.getFittest()
        return fittest

class GAMultiTSP(AlgoMultiTSP):
    def __init__(self, n_drones: int, nodes, labels=None):
        super().__init__("ga", n_drones, nodes, labels)
        self.pop = None

    def solve(self, numGenerations=70, mutationRate=0.02, tournamentSize=10, populationSize=100, elitism=True, cont=False):
        random.seed(int.from_bytes(os.urandom(8), 'big'))

        for node in self.network.nodes:
            RouteManager.addDustbin(Dustbin(x=node.x, y=node.y, z=node.z, label=node.label))

        if not cont or self.pop is None:
            self.cost_hist = []
            self.pop = Population(populationSize, True, len(self.network.nodes), self.n_drones)
        
        globalRoute: Route = self.pop.getFittest()
        print ('Initial minimum score: ' + str(globalRoute.getScore()))

        self.cost_hist.append(globalRoute.getScore())

        # Start evolving
        # pbar = progressbar.ProgressBar()
        for _ in tqdm(range(numGenerations), desc="GA Progress"):
            self.pop = GA.evolvePopulation(self.pop, len(self.network.nodes), self.n_drones, elitism, mutationRate, tournamentSize)
            localRoute: Route = self.pop.getFittest()
            self.cost_hist.append(localRoute.getScore())
            if globalRoute.getScore() > localRoute.getScore():
                globalRoute: Route = localRoute

        self.convert_to_network(globalRoute)

    def convert_to_network(self, route: Route) -> None:
        self.network.init_paths()

        for i, drone_route in enumerate(route.route):
            for dustbin in drone_route:
                curr_node: Optional[Node] = self.network.get_node_by_label(dustbin.label)
                if curr_node is not None and not self.network.is_start(curr_node):
                    self.network.add_node_to_path(i, curr_node)

    