from NetClassifier import Net
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import DataPreprocess

if __name__ == '__main__':
    # Step 1: Load and preprocess the data
    X_train, X_test, y_train, y_test = DataPreprocess.get_data()

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

    # Per label evaluation
    for i in range(output_size):
        print(f'Label {i+1}:')
        print(f'Accuracy: {accuracy_score(y_test[:, i], y_pred[:, i])}')
        print(f'Precision: {precision_score(y_test[:, i], y_pred[:, i])}')
        print(f'Recall: {recall_score(y_test[:, i], y_pred[:, i])}')
        print(f'F1 Score: {f1_score(y_test[:, i], y_pred[:, i])}')
        print()
