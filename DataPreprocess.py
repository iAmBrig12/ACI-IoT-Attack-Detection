import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('ACI-IoT-2023.csv')

remove_cols = ['Flow ID', 'Src IP', 'Dst IP', 'Timestamp', 'Label']

X = df.drop(columns=remove_cols)
y = df['Label']

y = pd.get_dummies(y, dtype=int)
X = pd.get_dummies(X, columns=['Connection Type'], dtype=int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69)

