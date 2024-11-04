import torch
import torch.nn as nn
import numpy as np  # Import numpy
from torch.utils.data import DataLoader, TensorDataset

class Net(nn.Module):
    def __init__(self, input_size, output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)  # Increased number of neurons
        self.bn1 = nn.BatchNorm1d(256)  # Batch normalization
        self.leaky_relu = nn.LeakyReLU()  # Changed activation function
        self.dropout = nn.Dropout(p=0.3)  # Increased dropout rate
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)  # Batch normalization
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)  # Batch normalization
        self.fc4 = nn.Linear(64, output_size)
        self.sigmoid = nn.Sigmoid()

        # Check for GPU availability 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using {self.device}')
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        x = self.fc4(x)
        x = self.sigmoid(x) 
        return x  
    
    def fit(self, X, y, num_epochs, learning_rate):
        print('Training the network...')
        class_weights = torch.tensor(1.0 / y.mean(axis=0), dtype=torch.float32).to(self.device)  # Adjust class weights calculation
        criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
        X = torch.tensor(X, dtype=torch.float32).to(self.device)  # Move data to device
        y = torch.tensor(y, dtype=torch.float32).to(self.device)  # Change dtype to float32

        self.to(self.device)  # Move model to device
        self.train()  # Set model to training mode
        
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            outputs = self(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            if (epoch+1) % (num_epochs/10) == 0:
                torch.cuda.synchronize()  # Synchronize for accurate timing
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        
        print('Training complete.')
        # Release CUDA tensors and clear cache
        del X, y
        torch.cuda.empty_cache()

    def predict(self, X):
        X = torch.tensor(X, dtype=torch.float32).to(self.device)  # Move data to device
        self.to(self.device)  # Move model to device

        self.eval()
        with torch.no_grad():
            predictions = torch.sigmoid(self.forward(X)).detach().cpu().numpy()
        return predictions