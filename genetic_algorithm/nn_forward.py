"""File with NN functions
"""
import numpy as np
from . import config as config

def sigmoid(x):
    '''Sigmoid function'''
    return 1.0 / (1.0 + np.exp(-x))  # sigmoid "squashing" function to interval [0,1]

def nn_forward(x, chromosome,hidden_neurons):
    '''Computing Neural Network output

       Parameters
       ----------
       x : ndarray
            inputs for NN
       chromosome : ndarray
            weights for NN
    
       Returns
       -------
       float
            a value [0, 1]

    '''
    input_size = config.INPUT_SIZE

    # weigths initialization from chromosome
    weights = []
    bias_weights = []
    last_length = 0

    # get each layer weights from chromosome
    for i in range(len(hidden_neurons)):
        w_length = 0
        if i == 0:
            w_length = hidden_neurons[0]*input_size+hidden_neurons[0]
        else:
            w_length = hidden_neurons[i]*hidden_neurons[i-1]+hidden_neurons[i]
        w = chromosome[last_length:last_length+w_length-hidden_neurons[i]]
        last_length += w_length
        if i == 0:
            w.shape = (hidden_neurons[i],input_size)
        else:
            w.shape = (hidden_neurons[i], hidden_neurons[i-1])
        weights.append(w)
        bias_array = chromosome[w_length-hidden_neurons[i]:w_length]
        bias_weights.append(bias_array)

    # get output layer weights
    w_out_length = hidden_neurons[-1]
    output_weigths = chromosome[last_length:]

    # weigths are ready to forward pass
    result = np.dot(weights[0], x)+bias_weights[0]
    result[result < 0] = 0 # ReLU nonlinearity
    for i in range(len(hidden_neurons)):
        if i > 0:
            result = np.dot(weights[i], result)+bias_weights[i]
            result[result < 0] = 0 # ReLU nonlinearity

    # forward pass after last layer
    logp = np.dot(output_weigths, result)  # log probability
    p = sigmoid(logp) #  probability of moving right. sigmoid nonlinearity squashes output to [0,1]

    return p