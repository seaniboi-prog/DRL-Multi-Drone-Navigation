# General

# ANSI escape codes for colors
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"
RESET = "\033[0m"

PLOT_COLOURS = ['g', 'b', 'orange', 'gold', 'c', 'm', 'y']

# Multi TSP

# HillClimb Constants
EPOCHS = 10000

# GA Constants
POPULATION_SIZE = 100
TOURNAMENT_SIZE = 10
GENERATIONS = 20
MUTATION_RATE = 0.05
CROSSOVER_RATE = 0.9
ELITISM = True

# ACO Constants
RHO = 0.03
Q = 1
ALPHA = 1
BETA = 3
GEN_SIZE = None
LIMIT = 200
OPT2 = 30

# Tabu Search Constants
MAX_TABU_SIZE = 10000
STOPPING_TURN = 500
NEIGHBOURHOOD_SIZE = 50

# DRL