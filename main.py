from NetClassifier import Net
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix  # Import confusion_matrix
import DataPreprocess

if __name__ == '__main__':
    # Step 1: Load and preprocess the data
    X_train, X_test, y_train, y_test = DataPreprocess.get_data()

    # Step 2: Initialize the network
    input_size = X_train.shape[1]
    output_size = y_train.shape[1]
    learning_rate = 0.001
    num_epochs = 10000

    net = Net(input_size, output_size)

    net.fit(X_train, y_train, num_epochs, learning_rate)

    # Step 3: Evaluate the network
    y_pred = net.predict(X_test)
    y_pred_binary = (y_pred > 0.5).astype(int)  # Convert predictions to binary

    # Per label evaluation
    labels = ['ARP Spoofing', 'Benign', 'DNS Flood', 'Dictionary Attack',
       'ICMP Flood', 'OS Scan', 'Ping Sweep', 'Port Scan', 'SYN Flood',
       'Slowloris', 'UDP Flood', 'Vulnerability Scan']
    for i in range(output_size):
        print(f'{labels[i]}:')
        print(f'Accuracy: {accuracy_score(y_test[:, i], y_pred_binary[:, i])}')
        print(f'Precision: {precision_score(y_test[:, i], y_pred_binary[:, i], zero_division=0)}')
        print(f'Recall: {recall_score(y_test[:, i], y_pred_binary[:, i], zero_division=0)}')
        print(f'F1 Score: {f1_score(y_test[:, i], y_pred_binary[:, i])}')
        print(f'Confusion Matrix:\n{confusion_matrix(y_test[:, i], y_pred_binary[:, i])}')
        print()

    # Overall evaluation
    print('Overall:')
    print(f'Accuracy: {accuracy_score(y_test, y_pred_binary)}')
    print(f'Precision: {precision_score(y_test, y_pred_binary, average="weighted", zero_division=0)}')
    print(f'Recall: {recall_score(y_test, y_pred_binary, average="weighted", zero_division=0)}')
    print(f'F1 Score: {f1_score(y_test, y_pred_binary, average="weighted")}')

