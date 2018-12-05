import tensorflow as tf
from tensorflow import keras
import numpy as np
import csv

'''
AUTHOR:
Arquilita, Jasper Ian Z.
CMSC 191-Z - Introduction to Neural Networks.

DESCRIPTION:
The model is a 1-3-1 neural network.

The layers are initialized with weights and biases of value 0.01.
The hidden layers uses the ReLU activation function,
    and the output layer use the sigmoid activation function.

There are 3 inputs:
    The number of days from January 1, 1990
    The recorded precipitation on that day
    The recorded air pressure on that day

DATASET:
ArcGIS Data Access Viewer. NASA. Retrieved from:
https://power.larc.nasa.gov/data-access-viewer/
'''

# constants
NUM_EPOCHS = 1000
RATIO_TEST = 0.2
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
    data = np.delete(data, 5, 1)[:, 4:6] # get the variables
    days = np.arange(data.shape[0]) # the number of days from Jan 1, 1990
    data = np.hstack((days.reshape(-1,1), data))

# Create the neural network model
model = keras.Sequential([
    # hidden layer 1
    keras.layers.Dense(
        data.shape[1], # number of nodes
        activation = tf.nn.relu,
        kernel_initializer = keras.initializers.Constant(value = 0.01),
        bias_initializer = keras.initializers.Constant(value = 0.01),
        input_shape = (data.shape[1],)), # number of inputs
    # hidden layer 2
    keras.layers.Dense(
        data.shape[1], # number of nodes
        activation = tf.nn.relu,
        kernel_initializer = keras.initializers.Constant(value = 0.01),
        bias_initializer = keras.initializers.Constant(value = 0.01)),
    # hidden layer 3
    keras.layers.Dense(
        data.shape[1], # number of nodes
        activation = tf.nn.relu,
        kernel_initializer = keras.initializers.Constant(value = 0.01),
        bias_initializer = keras.initializers.Constant(value = 0.01)),
    # output layer
    keras.layers.Dense(1, activation = keras.activations.sigmoid,
        kernel_initializer = keras.initializers.Constant(value = 0.01),
        bias_initializer = keras.initializers.Constant(value = 0.01))
])

# use the Gradient Descent with the specified learning rate
optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
# calculate MSE (Mean Squared Error) for loss and MAE (Mean Absolute Error) for evaluation
model.compile(loss = 'mse', optimizer = optimizer, metrics = ['mae'])
# model.summary() # used to print the model statistics

# callback to see the progress of the model
class EpochUpdate(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        print("EPOCH {}: LOSS = {:1.8f} ERROR = {:1.4f}%".format(epoch + 1, logs['loss'], logs['mean_absolute_error'] * 100))

# Perform training
# For cross validation, the data are split to training and test datas
# using the RATIO_TEST validation split, which is the percent of data to be used as test
history = model.fit(data, trueval, epochs = NUM_EPOCHS, validation_split = RATIO_TEST, verbose=0, callbacks = [EpochUpdate()])

# Now we can predict
print("=================================================================")
