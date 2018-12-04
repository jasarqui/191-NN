import numpy as np
import csv

'''
AUTHOR:
Arquilita, Jasper Ian Z.
CMSC 191-Z - Introduction to Neural Networks.

DESCRIPTION:
A 1-2-1 Neural Network that predicts the 
Relative Humidity on a certain day.

DATASET:
ArcGIS Data Access Viewer. NASA. Retrieved from:
https://power.larc.nasa.gov/data-access-viewer/
'''

# constants
NUM_NEURON = 2
NUM_INPUTS = 2
NUM_EPOCHS = 10
NUM_MCHEPS = 0.01
LEARNING_RATE = 0.01

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
    trueval = np.transpose(trueval) # change vector to by row
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
    return np.where(x <= 0, 0, 1)

# Forward Pass
def forward_pass(data, weights, bias, layer, is_output):
    if is_output: return sigmoid(np.matmul(output_weights, data) + bias)
    else:
        result = np.matmul(np.transpose(weights), data)
        return relu(np.matmul(np.transpose(weights), data))

# Backward Propagation
def backward_prop(data, hidden_weights, output_weights, bias, output_bias, hidden_layer_outputs, trueval, result):
    # the effect of output to error
    error_output = result - trueval
    error_output_delta = error_output * dsigmoid(result)

    # the effect of hidden layer 2 to error
    error_layer2 = np.matmul(error_output_delta.reshape(-1,1), np.transpose(hidden_weights)[1].reshape(1,-1))
    error_layer2_delta = np.transpose(error_layer2) * drelu(hidden_layer_outputs[1])

    # the effect of hidden layer 1 to error
    error_layer1 = np.matmul(np.transpose(error_layer2_delta), np.transpose(hidden_weights)[0])
    error_layer1_delta = np.transpose(error_layer1) * drelu(hidden_layer_outputs[0])

    # adjust the weights
    hidden_weights[0] += LEARNING_RATE * np.matmul(data, error_layer1_delta)
    hidden_weights[1] += LEARNING_RATE * np.matmul(np.transpose(hidden_layer_outputs[0]), np.transpose(error_layer2_delta))
    output_weights += LEARNING_RATE * np.matmul(np.transpose(hidden_layer_outputs[1]), np.transpose(error_output_delta))

    # adjust bias
    bias[0] -= LEARNING_RATE * np.matmul(data, error_layer1_delta)
    bias[1] -= LEARNING_RATE * np.matmul(np.transpose(hidden_layer_outputs[0]), np.transpose(error_layer2_delta))
    output_bias -= LEARNING_RATE * np.matmul(np.transpose(hidden_layer_outputs[1]), np.transpose(error_output_delta))

    return (hidden_weights, output_weights, bias, output_bias)

# Initialize Variables
# for loop checking
curr_epoch = 0
loss = 100

# Perform Training
# ends if max epochs is reached or tolerance is reached
while (curr_epoch < NUM_EPOCHS and loss > NUM_MCHEPS):
    is_tolerable = True # true until proven false    
    hidden_layer_outputs = np.array([])
    result = data

    print(hidden_weights)
    print(output_weights)

    for layer in (0, NUM_NEURON):
        # this will let the variables pass through the neurons
        # as it changes value
        if layer != NUM_NEURON:
            result = forward_pass(result, hidden_weights, np.transpose(bias)[layer], layer, False)
            # get the values of the hidden layer for backward propagation
            if layer == 0: hidden_layer_outputs = result
            else: hidden_layer_outputs = np.vstack((hidden_layer_outputs, result), 0)
    result = forward_pass(result, output_weights, output_bias, layer, True) # output layer
    
    print(result)

    # calculate loss
    error = result - trueval
    loss = np.sum(0.5 * (error ** 2))
    print("EPOCH {} LOSS {}".format(curr_epoch, loss))

    # perform backward propagation
    (hidden_weights, output_weights, bias, output_bias) = backward_prop(data, hidden_weights, output_weights, bias, output_bias, hidden_layer_outputs, trueval, result)
    
    # update
    curr_epoch += 1

print("Training done.")
print("Final Weights:")
print(hidden_weights)
print(output_weights)

# # Prediction
# choice = 0
# print("Predict the relative humidity at location {} {}".format(latitude, longitude))
# while(choice != 1):
#     precip = float(input("Enter precipitation: "))
#     prssre = float(input("Enter pressure: "))
    
#     # compute
#     new_data = np.array((precip, prssre))
#     # forward pass the new data into the neural network
#     result = new_data
#     for layer in (0, NUM_NEURON):
#         if layer != NUM_NEURON:
#             result = forward_pass(result, hidden_weights, np.transpose(bias)[layer], layer, False)
#     out = forward_pass(result, output_weights, output_bias, layer, True) * 100 # output layer 
#     print("The relative humidity at {} precipitation and {} pressure is {}".format(precip, prssre, out))

#     while(choice != 0 or choice != 1):
#         choice = int(input("Predict again?\n[0] Yes\n[1] No\n>> "))
#         if choice == 0: break
#         elif choice == 1:
#             print("Exiting..")
#             break
#         else: print("Invalid option.")
