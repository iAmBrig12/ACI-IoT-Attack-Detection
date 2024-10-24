import torch
import pandas as pd
from NetClassifier import Net
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

if __name__ == '__main__':
    # Load data
    X_train = pd.read_csv('X_train.csv').values
    X_test = pd.read_csv('X_test.csv').values
    y_train = pd.read_csv('y_train.csv').values
    y_test = pd.read_csv('y_test.csv').values

    # Convert NumPy arrays to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)

    # Step 2: Initialize the network
    input_size = X_train.shape[1]
    output_size = y_train.shape[1]
    batch_size = 100000
    learning_rate = 0.01
    num_epochs = 100

    net = Net(input_size, output_size)

    net.fit(X_train, y_train, num_epochs, learning_rate, batch_size)

    # Step 3: Evaluate the network
    y_pred = net.predict(X_test)

    # Convert predictions to NumPy arrays if necessary
    y_pred = y_pred.detach().numpy()

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
