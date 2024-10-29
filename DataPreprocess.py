import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def process_data():
    df = pd.read_csv('ACI-IoT-2023.csv')

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(axis='rows', inplace=True)

    remove_cols = ['Flow ID', 'Src IP', 'Dst IP', 'Timestamp', 'Label']

    X = df.drop(columns=remove_cols)
    y = df['Label']

    y = pd.get_dummies(y, dtype=int)
    X = pd.get_dummies(X, columns=['Connection Type'], dtype=int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=69)

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