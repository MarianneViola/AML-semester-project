from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint, uniform
import pandas as pd
import numpy as np

# Import data
filename = r"C:\Users\flyve\PycharmProjects\AML_shared\heart_failure_clinical_records_dataset.csv"
data = pd.read_csv(filename)

# Create test and train data
X, y = data.drop(['DEATH_EVENT'], axis=1), data['DEATH_EVENT']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=3)

# define the parameter distributions to search over
param_dist = {'n_estimators': np.arange(75, 300, 15),
              'max_features': np.arange(1, data.shape[1], 1),
              'max_depth': np.append(np.arange(2, 20, 2), None),
              'max_leaf_nodes': np.append(np.arange(2, 50, 2), None),
              'min_samples_split': np.append(np.arange(2, 20, 2), None),
              'min_impurity_decrease': uniform(0, 0.5),
              'bootstrap': [True, False]}

# create a random forest classifier
rf = RandomForestClassifier()

# define the random search with 10-fold cross-validation
random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_dist,
                                   n_iter=100, cv=10, verbose=2, random_state=3)

# fit the random search to the data
random_search.fit(X, y)

# print the best hyperparameters found by the random search
print("Best hyperparameters: ", random_search.best_params_)