import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import RobustScaler
import seaborn as sns
from sklearn.pipeline import Pipeline

# Import data
filename = r"C:\Users\flyve\PycharmProjects\AML_shared\heart_failure_clinical_records_dataset.csv"
data = pd.read_csv(filename)

# Create test and train data
x,y = data.drop(['DEATH_EVENT'], axis=1), data['DEATH_EVENT']
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=3)

# Make parameter dictionary for the grid search
param_grid_rf = {'max_leaf_nodes': np.arange(2, 50, 2)}
svm = Pipeline([
    ('scaler', RobustScaler()),
    ('SVM', SVC(kernel='linear', probability=True, C=10, gamma = 0.01, degree = 2, coef0 = 0))
])
# Create the grid and fit the train data into it
grid = GridSearchCV(svm, param_grid=param_grid_rf, cv=5, return_train_score=True, n_jobs=-1)

grid.fit(X_train, y_train)

results = pd.DataFrame(grid.cv_results_)

results.plot('param_max_leaf_nodes', 'mean_train_score')
results.plot('param_max_leaf_nodes', 'mean_test_score', ax=plt.gca()) # Tror "validation" score, but called "test" score; like Rahman has written.
plt.fill_between(results.param_max_leaf_nodes.astype(np.int),
                 results['mean_train_score'] + results['std_train_score'],
                 results['mean_train_score'] - results['std_train_score'], alpha=0.2)
plt.fill_between(results.param_max_leaf_nodes.astype(np.int),
                 results['mean_test_score'] + results['std_test_score'],
                 results['mean_test_score'] - results['std_test_score'], alpha=0.2)
plt.legend()

plt.show()

# Print results
print("best mean cross-validation score: {:.3f}".format(grid.best_score_))
print("best parameters: {}".format(grid.best_params_))
print("test-set score: {:.3f}".format(grid.score(X_test, y_test)))


# Create a DataFrame from the cv_results_
results_df = pd.DataFrame(grid.cv_results_)
pivot_table = pd.pivot_table(results_df, values='mean_test_score', index='param_n_estimators', columns='param_max_features')
sns.heatmap(pivot_table, annot=True, cmap='YlGnBu')


# Add labels and title
plt.xlabel('Max Features')
plt.ylabel('Number of Trees')
plt.title('Random Forest Grid Search Results')
plt.show()



