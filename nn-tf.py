import tensorflow as tf
from tensorflow import keras
from datetime import datetime
import time
import numpy as np
import csv
import matplotlib.pyplot as plt

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
BASE_DATE = datetime.strptime("01/01/1990", "%m/%d/%Y")
NUM_EPOCHS = 1000
LEARNING_RATE = 0.01
VALIDATION_RATIO = 0.22222
# Note that the validation ratio is adjusted to roughly 22%, 
# because we are taking from the training dataset which is only 90% of the original

# get dataset
with open('weather_data.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    data = np.array([])

    # parse dataset
    for row in csv_reader:
        if len(data) == 0: data = np.array(row).astype(float)
        else: data = np.vstack((data, np.array(row).astype(float)))
        
    # parse the dataset
    latitude, longitude = data[0,0], data[0,1] # get coordinates
    trueval = data[:, 5] / 100 # get the true values and normalize
    data = np.delete(data, 5, 1)[:, 4:6] # get the variables
    days = np.arange(data.shape[0]) # the number of days from Jan 1, 1990
    data = np.hstack((days.reshape(-1,1), data)) # add the days to the variables

    # get the test dataset
    test_length = round(0.1 * data.shape[0]) # gets the number of test data needed
    test = data[data.shape[0]-test_length : data.shape[0]] # gets the last 10% of the data
    data = np.delete(data, np.s_[data.shape[0]-test_length : data.shape[0]], 0) # remove the test data from the training dataset
    test_trueval = trueval[trueval.shape[0]-test_length : trueval.shape[0]] # get the last 10% of the true values
    trueval = np.delete(trueval, np.s_[trueval.shape[0]-test_length : trueval.shape[0]]) # remove the test true values from training true values

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
# using the VALIDATION_RATIO validation split, which is the percent of data to be used as validation
start_time = time.time()
history = model.fit(data, trueval, epochs = NUM_EPOCHS, validation_split = VALIDATION_RATIO, verbose = 0, callbacks = [EpochUpdate()])
end_time = time.time()
print("\nThe neural network performed {} epochs within {:.2f} seconds during training.".format(NUM_EPOCHS, end_time - start_time))
print("It was trained on a total of {} data split between {} training data and {} test data.".format(data.shape[0], data.shape[0] - int(data.shape[0] * VALIDATION_RATIO), int(data.shape[0] * VALIDATION_RATIO)))

# show the history plots to see results
def plot_history(hist):
    # create the epoch array
    epoch = (np.arange(len(hist['val_loss'])) + 1).tolist()

    # create the MAE plot
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error')
    plt.plot(epoch, hist['mean_absolute_error'], label='Train Error')
    plt.plot(epoch, hist['val_mean_absolute_error'], label = 'Val Error')
    plt.legend()
    plt.ylim([0,0.5])
    plt.show()
  
    # create the MSE plot
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.plot(epoch, hist['loss'], label='Train Error')
    plt.plot(epoch, hist['val_loss'], label = 'Val Error')
    plt.legend()
    plt.ylim([0,0.2])
    plt.show()

# plot
plot_history(history.history)

# Perform QA using test dataset
loss, mae = model.evaluate(test, test_trueval, verbose = 0)
print("Test Dataset Accuracy")
print("MAE: {:5.4f}".format(mae))
print("MSE: {:5.4f}".format(loss))

# Perform forward passes to test data and check results
test_predictions = model.predict(test).flatten()

# create a scatter plot of the results
plt.scatter(test_trueval, test_predictions)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.axis('equal')
plt.axis('square')
plt.xlim([0, plt.xlim()[1]])
plt.ylim([0, plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])
plt.show()

# plot the error bar graph
error = test_predictions - test_trueval
plt.hist(error, bins = 25)
plt.xlabel("Prediction Error")
_ = plt.ylabel("Count")
plt.show()

# # Now we can predict
# choice = 0
# print("=================================================================\n")
# print("Predict the relative humidity at location ({}, {}).".format(latitude, longitude))

# while choice != 1:
#     # get the data
#     date = input("Enter which date you wish to predict the weather (MM/DD/YYYY): ")
#     date_parsed = datetime.strptime(date, "%m/%d/%Y")
#     precip = float(input("Enter precipitation: "))
#     prssre = float(input("Enter pressure: "))
#     # the added [0,0,0] is padding, tensorflow does not allow 1-D vector as input even if reshaped
#     data = np.array([[0,0,0],[(date_parsed - BASE_DATE).days, precip, prssre]])

#     # predict
#     prediction = model.predict(data).flatten()[1]
#     print("\nThe relative humidity on {} at {}mm precipitation and {}kPa pressure is {:.2f}%".format(date, precip, prssre, prediction * 100))

#     while(choice != 0 or choice != 1):
#         choice = int(input("\nPredict again?\n[0] Yes\n[1] No\n>> "))
#         if choice == 0: 
#             print("")
#             break
#         elif choice == 1:
#             print("Exiting..")
#             break
#         else: print("Invalid option.")
