from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from NetClassifier import Net
import DataPreprocess
import pandas as pd

# Define the parameter grid
param_grid = {
    'hidden_sizes': [[5], [10], [5, 5], [10, 5]],  # Example hidden layer sizes
    'learning_rate': [0.01, 0.001],
    'num_epochs': [100, 200]
}

X_train = pd.read_csv('X_train.csv').to_numpy()
y_train = pd.read_csv('y_train.csv').to_numpy()
X_test = pd.read_csv('X_test.csv').to_numpy()
y_test = pd.read_csv('y_test.csv').to_numpy()

# Create a GridSearchCV object
grid_search = GridSearchCV(Net(), param_grid, cv=3, scoring='accuracy')

grid_search.fit(X_train, y_train)  # Fit the model

print(grid_search.best_params_)  # Print the best parameters
print(grid_search.best_score_)  # Print the best accuracy

best_model = grid_search.best_estimator_  # Get the best model
y_pred = best_model.predict(X_test)  # Predict on the test set

accuracy = accuracy_score(y_test, y_pred)  # Compute the accuracy
print(accuracy)  # Print the accuracy