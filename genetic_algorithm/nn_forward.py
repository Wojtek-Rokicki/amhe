"""File with NN functions
"""
import numpy as np
from . import config as config

def sigmoid(x):
    '''Sigmoid function'''
    return 1.0 / (1.0 + np.exp(-x))  # sigmoid "squashing" function to interval [0,1]

def nn_forward(x, chromosome): # takes the state as input
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

    hidden_neurons = config.hidden_neurons
    input_size = config.INPUT_SIZE

    # weigths initialization from chromosome
    weights = []
    bias_weights = []
    last_length = 0
    # TODO: make it cleaner
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

    # output layer
    w_out_length = hidden_neurons[-1]
    output_weigths = chromosome[last_length:]

    # weigths are ready to forward pass
    result = np.dot(weights[0], x)+bias_weights[0]
    result[result < 0] = 0 # ReLU nonlinearity, could switch to smthg else later (tanh, sigmoid, etc..)
    for i in range(len(hidden_neurons)):
        if i > 0:
            result = np.dot(weights[i], result)+bias_weights[i]
            result[result < 0] = 0 # ReLU nonlinearity, TODO: put it to another function

    # forward pass after last layer
    logp = np.dot(output_weigths, result)  # log probability
    p = sigmoid(logp) #  probability of moving right. sigmoid nonlinearity squashes output to [0,1]


    # # condition the chromosome into matrix shape to do neural net forward pass
    # # chromosome.shape = (hidden_neurons+1, input_size)  # + 1 to include the row of output weights from hidden layer to output
    # w1_length = hidden_neurons * input_size
    # w1 = chromosome[0:w1_length]
    # w1.shape = (hidden_neurons, input_size)
    # w2 = chromosome[w1_length+1:-1]

    # #  w1 and w2 now ready to do forward pass
    # h = np.dot(w1, x)+chromosome[w1_length]
    # h[h < 0] = 0  # ReLU nonlinearity, could switch to smthg else later (tanh, sigmoid, etc..)
    # logp = np.dot(w2, h) + chromosome[-1]  # log probability
    # p = sigmoid(logp) #  probability of moving right. sigmoid nonlinearity squashes output to [0,1]
    return p