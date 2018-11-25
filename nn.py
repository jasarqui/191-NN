import numpy as np
import torch
import csv

# constants
NUM_NEURON = 3
NUM_INPUTS = 2

# get dataset
with open('weather_data.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    data = np.array([])

    # parse dataset
    for row in csv_reader:
        if len(data) == 0: data = np.array(row).astype(float)
        else: data = np.vstack((data, np.array(row).astype(float)))
        
    latitude, longitude = data[0,0], data[0,1] # get coordinates
    trueval = data[:, 5] # get the true value
    data = np.delete(data, 5, 1)[:, 4:6] # get the variables

# initial memory
weights = torch.from_numpy(np.zeros((NUM_NEURON, NUM_INPUTS)))
bias = torch.from_numpy(np.zeros((NUM_NEURON, NUM_INPUTS)))

# Forward Pass
def forward_pass(data, weights, bias, layer):
    return torch.relu(torch.matmul(torch.t(weights[layer]), data) + bias)

# midterm exam
# w = torch.from_numpy(np.array([0.16, 0.73]))
# a0 = torch.from_numpy(np.array([0.33, 0.59]))
# a = torch.from_numpy(np.array([[0.01, 0.44], [0.03, 0.23]]))
# print(forward_pass(w, a, a0, 1))

# Backward Propagation
