import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline

# Import data
filename = r"C:\Users\flyve\PycharmProjects\AML_shared\heart_failure_clinical_records_dataset.csv"
data = pd.read_csv(filename)

# Create test and train data
x,y = data.drop(['DEATH_EVENT'], axis=1), data['DEATH_EVENT']
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=3)

# Make parameter dictionary for the grid search
param_grid_rf = {'SVM__gamma': np.arange(0.1, 6.00, 0.1)}
svm = Pipeline([
    ('scaler', RobustScaler()),
    ('SVM', SVC(kernel='linear', probability=True, C=10, degree = 2, coef0 = 0))
])
# Create the grid and fit the train data into it
grid = GridSearchCV(svm, param_grid=param_grid_rf, cv=5, return_train_score=True, n_jobs=-1)

grid.fit(X_train, y_train)

results = pd.DataFrame(grid.cv_results_)

# Plot the mean test score and the mean train score as a function of gamma
plt.plot(param_grid_rf['SVM__gamma'], results['mean_test_score'], label='Test Score')
plt.plot(param_grid_rf['SVM__gamma'], results['mean_train_score'], label='Train Score')
plt.xlabel('Gamma')
plt.ylabel('Score')
plt.legend()
plt.show()