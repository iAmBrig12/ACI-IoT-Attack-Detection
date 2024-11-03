import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def process_data():
    df = pd.read_csv('ACI-IoT-2023.csv')

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(axis='rows', inplace=True)

    remove_cols = ['Flow ID', 'Src IP', 'Dst IP', 'Timestamp', 'Label']

    X = df.drop(columns=remove_cols)
    y = df['Label']

    y = pd.get_dummies(y, dtype=int)
    X = pd.get_dummies(X, columns=['Connection Type'], dtype=int)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=69)

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