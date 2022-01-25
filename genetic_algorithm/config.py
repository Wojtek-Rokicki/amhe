"""Config file with parameters for GA and NN
"""

# Genetic Algorithm parameters
GENERATIONS = 4 # const
population_size = 10 
crossed_parents = 100 # number of chromosomes, that go through crossover
crossover_rate = 0.6 
mutation_rate = 0.1 
mutation_variation = 1

# Neural Network parameters
INPUT_SIZE = 4  # 4 states (information from environment) as the inputs to the NN
hidden_neurons = [2,2] # list of numberes of neurons in hidden layers (without bias)
# hidden_neurons = 2 