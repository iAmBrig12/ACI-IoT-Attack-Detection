import torch
import torch.nn as nn
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class Net(BaseEstimator, ClassifierMixin):
    def __init__(self, input_size, output_size, hidden_sizes):
        super(Net, self).__init__()
        
        layers = []

        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())
        
        for i in range(1, len(hidden_sizes)):
            layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_sizes[-1], output_size))

        self.model = nn.Sequential(*layers)

    def fit(self, X, y):
        # Convert data to PyTorch tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)

        # Initialize the model, loss function, and optimizer
        self.model = Net(input_size=X.shape[1], output_size=len(np.unique(y)), hidden_sizes=self.hidden_sizes)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Training loop
        for epoch in range(self.num_epochs):
            optimizer.zero_grad()  # Zero gradients
            outputs = self.model(X_tensor)  # Forward pass
            loss = criterion(outputs, y_tensor)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights

        return self

    def predict(self, X):
        # Convert data to PyTorch tensor
        X_tensor = torch.FloatTensor(X)
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs.data, 1)  # Get predicted class
        return predicted.numpy()

