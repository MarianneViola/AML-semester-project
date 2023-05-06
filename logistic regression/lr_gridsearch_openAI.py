import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
import numpy as np
from sklearn.preprocessing import RobustScaler

# Load the dataset
filename = r"C:\Users\flyve\PycharmProjects\AML_shared\heart_failure_clinical_records_dataset.csv"
heart_df = pd.read_csv(filename)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    heart_df.drop('DEATH_EVENT', axis=1), heart_df['DEATH_EVENT'],
    test_size=0.2, random_state=3
)

# scale
scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)

# Define the hyperparameter space to search over
param_dist = {'penalty': ['l1', 'l2', 'elasticnet', 'none'],
              'C': np.logspace(-3, 3, 7),
              'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
              'max_iter': [100, 500, 1000]}

# Create a logistic regression model
lr = LogisticRegression()

# Create a random search object
grid = GridSearchCV(lr, param_grid=param_dist, cv=5)

# Fit the model to the training data
grid.fit(X_train, y_train)

# Print the best hyperparameters and score
print("Best hyperparameters: ", grid.best_params_)
print("Best score: ", grid.best_score_)
