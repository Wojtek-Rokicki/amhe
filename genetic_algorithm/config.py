# HYPERPARAMETERS
GENERATIONS = 2  # max number of generations
population_size = 100  # number of chromosomes in each population

hidden_neurons = [2,2] # list of numberes of neurons in hidden layers (without bias)
# hidden_neurons = 2 

input_size = 4  # 4 states as the inputs to the NN
Pc = 0.6  # crossover rate
Pm = 0.1  # mutation rate
crossed_parents = 10  # number of chromosomes that go through crossover, needs to be even
