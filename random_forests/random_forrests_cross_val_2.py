from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import pandas as pd

# Import data
filename = r"C:\Users\flyve\PycharmProjects\AML_shared\heart_failure_clinical_records_dataset.csv"
data = pd.read_csv(filename)

# Create test and train data
X, y = data.drop(['DEATH_EVENT'], axis=1), data['DEATH_EVENT']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=3)

################################# random forests #########################
# make random forrest classifier
rfc = RandomForestClassifier(random_state=3,n_estimators=100, max_features=3)
rfc.fit(X_train, y_train)


# define the parameter grid for grid search
param_grid = {
    'n_estimators': [100, 200, 300],  # number of trees in the forest
    'max_depth': [5, 10, 15],  # maximum depth of each tree
    'min_samples_split': [2, 5, 10],  # minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4]  # minimum number of samples required to be at a leaf node
}

# create a random forest classifier
rf = RandomForestClassifier(random_state=3, max_features='sqrt')

# create a grid search object
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy')

# fit the grid search object to the data
grid_search.fit(X, y)

# print the best parameters and score found by grid search
print("Best Parameters: ", grid_search.best_params_)
print("Best Accuracy: ", grid_search.best_score_)