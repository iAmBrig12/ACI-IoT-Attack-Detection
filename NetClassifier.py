import torch
import torch.nn as nn
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class Net(BaseEstimator, ClassifierMixin):
    def __init__(self, input_size, hidden_sizes, output_size, learning_rate=0.001, num_epochs=100):
        super(Net, self).__init__()
        
        self.hidden_sizes = hidden_sizes
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        
        layers = []

        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())
        
        for i in range(1, len(hidden_sizes)):
            layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        layers.append(nn.Sigmoid())  # Use Sigmoid for multi-label classification

        self.model = nn.Sequential(*layers)

    def fit(self, X, y):
        # Convert data to PyTorch tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)  # Ensure y_tensor is of type Float

        # Initialize the loss function and optimizer
        criterion = nn.BCELoss()  # Use BCELoss for multi-label classification
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Training loop
        for epoch in range(self.num_epochs):
            optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()

    def predict(self, X):
        # Convert data to PyTorch tensor
        X_tensor = torch.FloatTensor(X)
        with torch.no_grad():
            outputs = self.model(X_tensor)
        return outputs.numpy()
