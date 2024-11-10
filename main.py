from NetClassifier import Net
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import DataPreprocess
import sys
import pandas as pd

save_eval = False
if len(sys.argv) > 1 and int(sys.argv[1]):
    save_eval = True
    eval_df = pd.DataFrame(columns=['Label', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'Confusion Matrix'])

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
        acc = accuracy_score(y_test[:, i], y_pred_binary[:, i])
        prec = precision_score(y_test[:, i], y_pred_binary[:, i], zero_division=0)
        rec = recall_score(y_test[:, i], y_pred_binary[:, i], zero_division=0)
        f1 = f1_score(y_test[:, i], y_pred_binary[:, i])
        cm = confusion_matrix(y_test[:, i], y_pred_binary[:, i])

        print(f'{labels[i]}:')
        print(f'Accuracy: {acc}')
        print(f'Precision: {prec}')
        print(f'Recall: {rec}')
        print(f'F1 Score: {f1}')
        print(f'Confusion Matrix:\n{cm}')
        print()

        if save_eval:
            eval_df = pd.concat([eval_df, pd.DataFrame([{'Label': labels[i], 'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1 Score': f1, 'Confusion Matrix': cm}])], ignore_index=True)

    # Overall evaluation
    acc = accuracy_score(y_test, y_pred_binary)
    prec = precision_score(y_test, y_pred_binary, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred_binary, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred_binary, average="weighted")

    print('Overall:')
    print(f'Accuracy: {acc}')
    print(f'Precision: {prec}')
    print(f'Recall: {rec}')
    print(f'F1 Score: {f1}')

    if save_eval:
        eval_df = pd.concat([eval_df, pd.DataFrame([{'Label': 'Overall', 'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1 Score': f1, 'Confusion Matrix': 'n/a'}])], ignore_index=True)
        eval_df.to_csv('evaluation.csv', index=False)

