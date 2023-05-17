import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import RobustScaler

# Load the dataset
filename = "heart_failure_clinical_records_dataset.csv"
data = pd.read_csv(filename)
data = data.drop(['serum_creatinine'], axis=1)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    data.drop('DEATH_EVENT', axis=1), data['DEATH_EVENT'],
    test_size=0.2, random_state=3)

# scale
scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)

# Define the hyperparameter space to search over
param_dist = {'penalty': ['l1', 'l2', 'elasticnet', 'None'],
              'dual': [True, False],
              'C': np.logspace(-3, 3, 7),
              'fit_intercept': [True, False],
              'intercept_scaling': np.logspace(-3, 3, 7),
              'class_weight': ['balanced', None],
              'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
              'max_iter': [100, 500, 1000],
              'multi_class': ['auto', 'ovr', 'multinomial']}


# Create a logistic regression model
lr = LogisticRegression()

# Create a random search object
random_search = RandomizedSearchCV(lr, param_distributions=param_dist, cv=10, return_train_score=True, random_state=3)

# Fit the model to the training data
random_search.fit(X_train, y_train)
results = pd.DataFrame(random_search.cv_results_)

# Print the best hyperparameters and score
print("Best hyperparameters: ", random_search.best_params_)
print("Best score: ", random_search.best_score_)