import numpy as np
import csv

'''
AUTHOR:
Arquilita, Jasper Ian Z.
CMSC 191-Z - Introduction to Neural Networks.

DESCRIPTION:
A 1-3-1 Neural Network that predicts the 
Relative Humidity on a certain day.

DATASET:
ArcGIS Data Access Viewer. NASA. Retrieved from:
https://power.larc.nasa.gov/data-access-viewer/
'''

# constants
NUM_NEURON = 3
NUM_INPUTS = 2
NUM_EPOCHS = 1000
NUM_MCHEPS = 0.01
BIAS_GAMMA = 0.5

# get dataset
with open('weather_data.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    data = np.array([])

    # parse dataset
    for row in csv_reader:
        if len(data) == 0: data = np.array(row).astype(float)
        else: data = np.vstack((data, np.array(row).astype(float)))
        
    latitude, longitude = data[0,0], data[0,1] # get coordinates
    trueval = data[:, 5] / 100 # get the true values and normalize
    data = np.delete(data, 5, 1)[:, 4:6] # get the variables
    data = np.transpose(data) # change variables by row

# Initial Memory
hidden_weights = np.zeros((NUM_INPUTS, NUM_NEURON)) + 0.1
output_weights = np.zeros(NUM_INPUTS) + 0.1
bias = np.zeros((NUM_INPUTS, NUM_NEURON)) + 0.1
output_bias = np.array([0.1])

# Activation Functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return x * (x > 0)

# Activation Function Derivatives
def dsigmoid(x):
    return x * (1 - x)

def drelu(x):
    if x <= 0: return 0
    else: return 1

# Forward Pass
def forward_pass(data, weights, bias, layer, is_output):
    if is_output: return sigmoid(np.matmul(np.transpose(weights)[layer], data) + bias)
    else:
        result = np.matmul(np.transpose(weights), data)
        return relu(np.matmul(np.transpose(weights), data))

# Backward Propagation
def backward_prop(data, hidden_weights, output_weights, bias, output_bias, hidden_layer_outputs, trueval, result):
    # the effect of output to error
    error_output = result - trueval
    error_output_delta = error * dsigmoid(result)

    # the effect of hidden layer 3 to error
    error_layer3 = np.matmul(error_output_delta, np.transpose(hidden_weights)[2])
    error_layer3_delta = error_layer3 * drelu(hidden_layer_outputs[2]) 

    # the effect of hidden layer 2 to error
    error_layer2 = np.matmul(error_layer3_delta, np.transpose(hidden_weights)[1])
    error_layer2_delta = error_layer2 * drelu(hidden_layer_outputs[1])

    # the effect of hidden layer 1 to error
    error_layer1 = np.matmul(error_layer2_delta, np.transpose(hidden_weights)[0])
    error_layer1_delta = error_layer1 * drelu(hidden_layer_outputs[0])

    # adjust the weights
    hidden_weights[0] += np.matmul(data, error_layer1_delta)
    hidden_weights[1] += np.matmul(np.transpose(hidden_layer_outputs[0]), error_layer2_delta)
    hidden_weights[2] += np.matmul(np.transpose(hidden_layer_outputs[1]), error_layer3_delta)
    output_weights += np.matmul(np.transpose(hidden_layer_outputs), error_output_delta)

    # adjust bias
    output_bias -= BIAS_GAMMA * error_output_delta
    bias[0] -= BIAS_GAMMA * error_layer1_delta
    bias[1] -= BIAS_GAMMA * error_layer2_delta
    bias[2] -= BIAS_GAMMA * error_layer3_delta

    return (hidden_weights, output_weights, bias, output_bias)

# Initialize Variables
# for loop checking
result = np.zeros(len(data[0]))
curr_epoch = 0
loss = 100

# Perform Training
# ends if max epochs is reached or tolerance is reached
while (curr_epoch < NUM_EPOCHS and loss > NUM_MCHEPS):
    is_tolerable = True # true until proven false

    for datum_index in (0, len(data[0])):
        hidden_layer_outputs = np.array([])
        result[datum_index] = data[datum_index] # initialize as the starting variable value

        for layer in (0, NUM_NEURON + 1): # + 1 to include the output layer
            # this will let the variables pass through the neurons
            # as it changes value
            if layer != NUM_NEURON:
                hidden_layer_outputs = forward_pass(result, hidden_weights, bias[layer], layer, False)
                if layer == 0: hidden_layer_outputs = result[datum_index]
                else: hidden_layer_outputs = np.vstack((hidden_layer_outputs, result[datum_index]), 0)
            else: result[datum_index] = forward_pass(result, output_weights, output_bias, layer, True) # output layer
    
    # calculate loss
    error = result - trueval
    loss = np.sum(0.5 * (error ** 2))
    print("EPOCH {} LOSS {}".format(curr_epoch, loss))

    # perform backward propagation
    (hidden_weights, output_weights, bias, output_bias) = backward_prop(data, hidden_weights, output_weights, bias, output_bias, hidden_layer_outputs, trueval, result)
    
    # update
    curr_epoch += 1

print("Training done.")