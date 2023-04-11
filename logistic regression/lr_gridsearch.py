from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import pandas as pd

# Import data
filename = r"C:\Users\flyve\PycharmProjects\AML_shared\heart_failure_clinical_records_dataset.csv"
data = pd.read_csv(filename)

# Create test and train data
x,y = data.drop(['DEATH_EVENT'], axis=1), data['DEATH_EVENT']
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=3)

# Create a pipeline with StandardScaler and LogisticRegression
pipe = make_pipeline(StandardScaler(), LogisticRegression())

# Define the hyperparameters to tune
param_grid = {'logisticregression__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
              'logisticregression__penalty': ['l1', 'l2', 'elasticnet']}

# Create a GridSearchCV object and fit it to the data
grid = GridSearchCV(pipe, param_grid=param_grid, cv=5)
grid.fit(X_test, y_test)

# Print the best hyperparameters and the corresponding mean cross-validation score
print("Best hyperparameters: ", grid.best_params_)
print("Best score: ", grid.best_score_)

