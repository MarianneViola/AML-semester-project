from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
import pandas as pd
from sklearn.preprocessing import RobustScaler

# Import data
filename = r"C:\Users\flyve\PycharmProjects\AML_shared\heart_failure_clinical_records_dataset.csv"
data = pd.read_csv(filename)

# Create test and train data
x, y = data[['time']], data['DEATH_EVENT']
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=3)

# Scale the data using RobustScaler
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the parameter grid for grid search
param_grid = {
    'kernel': ['linear', 'poly', 'rbf'],
    'C': [0.01, 0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1, 10],
    'degree': [2, 3, 4],
}

# Perform grid search using cross-validation
svm = SVC()
grid_search = GridSearchCV(svm, param_grid, cv=5)
grid_search.fit(X_train_scaled, y_train)

# Print the best hyperparameters and test accuracy
print("Best hyperparameters:", grid_search.best_params_)
test_acc = grid_search.score(X_test_scaled, y_test)
print("Test accuracy:", test_acc)