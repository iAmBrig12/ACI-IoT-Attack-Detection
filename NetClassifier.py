import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class Net(nn.Module):
    def __init__(self, input_size, output_size):
        super(Net, self).__init__()
        self.fc = nn.Linear(input_size, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)  # Dropout with 50% probability
        self.h = nn.Linear(32, output_size)
        self.sigmoid = nn.Sigmoid()

        # Check for GPU availability 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using {self.device}')
    
    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.h(x)
        x = self.sigmoid(x)
        return x
    
    def fit(self, X, y, num_epochs, learning_rate, batch_size=256):
        print('Training the network...')
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        # Check for GPU availability and move model to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        
        # Create TensorDataset and DataLoader with multiple workers
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(num_epochs):
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)  # Move data to the same device as the model
                
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        
        print('Training complete.')

    def predict(self, X):
        X = torch.tensor(X, dtype=torch.float32)
        self.to(self.device)

        self.eval()
        with torch.no_grad():
            return self.forward(X).detach().numpy()