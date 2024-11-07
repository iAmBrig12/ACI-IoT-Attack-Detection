import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

def process_data():
    df = pd.read_csv('ACI-IoT-2023.csv')

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(axis='rows', inplace=True)

    remove_cols = ['Flow Bytes/s', 'Flow Packets/s', 'Flow ID', 'Src IP', 'Dst IP', 'Timestamp', 'Label']

    X = df.drop(columns=remove_cols)
    y = df['Label']

    y = pd.get_dummies(y, dtype=int)
    X = pd.get_dummies(X, columns=['Connection Type'], dtype=int)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=69)
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Convert numpy arrays back to DataFrames
    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)
    y_train = pd.DataFrame(y_train)
    y_test = pd.DataFrame(y_test)

    X_train.to_csv('X_train.csv', index=False)
    X_test.to_csv('X_test.csv', index=False)
    y_train.to_csv('y_train.csv', index=False)
    y_test.to_csv('y_test.csv', index=False)

def get_data():
    X_train = pd.read_csv('X_train.csv').values
    X_test = pd.read_csv('X_test.csv').values
    y_train = pd.read_csv('y_train.csv').values
    y_test = pd.read_csv('y_test.csv').values

    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    process_data()