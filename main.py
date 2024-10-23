import DataPreprocess
import NetClassifier
import torch
import pandas as pd

# Step 1: Load and preprocess data
X_train = pd.read_csv('X_train.csv').values  # Convert DataFrame to NumPy array
X_test = pd.read_csv('X_test.csv').values  # Convert DataFrame to NumPy array
y_train = pd.read_csv('y_train.csv').values  # Convert DataFrame to NumPy array
y_test = pd.read_csv('y_test.csv').values  # Convert DataFrame to NumPy array

# Step 2: Initialize the network
input_size = X_train.shape[1]
hidden_sizes = [64, 32]  # Example hidden layer sizes, adjust as needed
output_size = y_train.shape[1]  # Number of output features
learning_rate = 0.001
num_epochs = 100

net = NetClassifier.Net(input_size, hidden_sizes, output_size, learning_rate, num_epochs)

# Step 3: Train the network
net.fit(X_train, y_train)
