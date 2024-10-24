import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class Net(nn.Module):
    def __init__(self, input_size, output_size):
        super(Net, self).__init__()
        self.fc = nn.Linear(input_size, 32)
        self.h = nn.Linear(32, output_size)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.fc(x)
        x = self.h(x)
        x = self.sigmoid(x)
        return x
    
    def fit(self, X, y, num_epochs, learning_rate, batch_size=256):
        print('Training the network...')
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
        # Check for GPU availability and move model to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        
        # Create TensorDataset and DataLoader
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(num_epochs):
            for inputs, targets in dataloader:
                # Move data to GPU if available
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs = self.forward(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

            if (epoch+1) % int(num_epochs/10) == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

    def predict(self, X):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        X = X.to(device)
        outputs = self(X)
        return torch.round(outputs)