import numpy as np
from . import config as config

def crossover(crossover_pairs):  # performs crossover on the current population

    population_size = config.population_size
    crossed_parents = config.crossed_parents

    selector = np.random.random_integers(1, population_size)  # used to select which parents to crossover from existing population
    selector = 1
    offspring = np.zeros([np.size(crossover_pairs, 0), np.size(crossover_pairs, 1)])
    for i in range(int(crossed_parents/2)):
        temp = crossover_pairs[2*i][selector:]
        temp2 = crossover_pairs[2*i+1][selector:]

        offspring[2*i][selector:] = temp2
        offspring[2*i+1][selector:] = temp

    for i in range(np.size(crossover_pairs, 0)):
        offspring[i][0] = crossover_pairs[i][0]

    return offspring