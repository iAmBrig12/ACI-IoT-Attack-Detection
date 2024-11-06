import torch
import torch.nn as nn
import time  # Import time module

class Net(nn.Module):
    def __init__(self, input_size, output_size):
        super(Net, self).__init__()

        # Define the network architecture
        self.fc1 = nn.Linear(input_size, 64)
        self.leaky_relu = nn.LeakyReLU()
        self.fc4 = nn.Linear(64, output_size)

        # Check for GPU availability 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using {self.device}')
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.leaky_relu(x)
        x = self.fc4(x) 
        return x  
    
    def fit(self, X, y, num_epochs, learning_rate):
        print('Training the network...')
        class_weights = torch.tensor(1.0 / y.mean(axis=0), dtype=torch.float32).to(self.device)  
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
        # move data to device
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        y = torch.tensor(y, dtype=torch.float32).to(self.device) 

        self.to(self.device)  
        self.train() 
        
        start_time = time.time()  # Start timer
        
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            outputs = self(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            if (epoch+1) % (num_epochs/10) == 0:
                torch.cuda.synchronize()  # Synchronize for accurate timing
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

        end_time = time.time() 
        print(f'Training completed in {end_time - start_time:.2f} seconds')

        # Release CUDA tensors and clear cache
        del X, y
        torch.cuda.empty_cache()

    def predict(self, X):
        # Move data and model to device
        X = torch.tensor(X, dtype=torch.float32).to(self.device) 
        self.to(self.device) 

        self.eval()
        with torch.no_grad():
            predictions = self.forward(X).detach().cpu().numpy()
        return predictions