from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint, uniform
import pandas as pd

# Import data
filename = r"C:\Users\flyve\PycharmProjects\AML_shared\heart_failure_clinical_records_dataset.csv"
data = pd.read_csv(filename)

# Create test and train data
X, y = data.drop(['DEATH_EVENT'], axis=1), data['DEATH_EVENT']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=3)

# define the parameter distributions to search over
param_dist = {'n_estimators': randint(10, 1000),
              'max_features': ['auto', 'sqrt', 'log2'],
              'max_depth': [None] + list(range(1, 50)),
              'max_leaf_nodes': [None] + list(range(2, 50)),
              'min_samples_split': randint(2, 20),
              'min_impurity_decrease': uniform(0, 0.5),
              'bootstrap': [True, False]}
print(param_dist)
# create a random forest classifier
rf = RandomForestClassifier()

# define the random search with 10-fold cross-validation
random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_dist,
                                   n_iter=100, cv=10, verbose=2, random_state=3)

# fit the random search to the data
random_search.fit(X, y)

# print the best hyperparameters found by the random search
print("Best hyperparameters: ", random_search.best_params_)