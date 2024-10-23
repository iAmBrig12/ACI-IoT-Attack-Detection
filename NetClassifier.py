import torch.nn as nn
import torch

class Net(nn.Module):
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

    def fit(self, X, y, epochs=1000, lr=0.01):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            y_pred = self.model(X)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()


    def forward(self, x):
        return self.model(x)