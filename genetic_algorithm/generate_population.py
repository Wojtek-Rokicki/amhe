"""File with GA functions
"""

import numpy as np
from . import config as config

def init_population():
    '''Generates initial population for GA using Xavier initialization
    
    Returns
    -------
    ndarray
        a population array with dimensions of (population, weights)
    '''
    hidden_neurons = config.hidden_neurons
    input_size = config.INPUT_SIZE
    population_size = config.population_size

    # need 10 initial chromosomes
    population = []
    print(f'Population size: {population_size}')
    # population.shape = (hidden_neurons+1, input_size)
    for j in range(population_size):
        # initialize the weights (with biases) using Xavier initialization

        weights = []
        for i in range(len(hidden_neurons)): # for each layer
            w_length = 0
            if i ==0:
                w_length = hidden_neurons[0]*input_size+hidden_neurons[0]
            else:
                w_length = hidden_neurons[i]*hidden_neurons[i-1]+hidden_neurons[i]
            w = np.random.randn(w_length) / np.sqrt(w_length) # weights for current layer
            w.shape = (1, w_length)
            weights.append(w) # list of ndarrays

        # one NN output
        # last layer
        w_out_length = hidden_neurons[-1]
        w_out = np.random.randn(w_out_length) / np.sqrt(w_out_length)
        w_out.shape = (1, w_out_length)
        weights.append(w_out)

        # w1_length = hidden_neurons * input_size +1
        # w1 = np.random.randn(w1_length) / np.sqrt(input_size)
        # w1.shape = (1, w1_length)
        # #tutaj powinna być po prostu ostatnia warstwa
        # w2 = np.random.randn(hidden_neurons+1) / np.sqrt(hidden_neurons+1)
        # w2.shape = (1, hidden_neurons+1)

        # a = np.concatenate((w1, w2), axis=1)
        a = np.concatenate(weights, axis=1)
        chromosome_length = len(a[0])
        if j == 0:
            population = a
        else:
            population = np.vstack((population, a))

    return population, chromosome_length