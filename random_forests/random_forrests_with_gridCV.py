from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import data
filename = 'heart_failure_clinical_records_dataset.csv'
data = pd.read_csv(filename)

# Create test and train data
x,y = data.drop(['DEATH_EVENT'], axis=1), data['DEATH_EVENT']
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=3)

# Make parameter dictionary for the grid search
param_grid_rf = {'max_features': np.arange(1, 9, 1),
                 'n_estimators': np.arange(75, 275, 15)}
rfc = RandomForestClassifier(random_state=3)

# Create the grid and fit the train data into it
grid = GridSearchCV(rfc, param_grid=param_grid_rf,
                    cv=5, return_train_score=True)

grid.fit(X_train, y_train)

# Print results
print("best mean cross-validation score: {:.3f}".format(grid.best_score_))
print("best parameters: {}".format(grid.best_params_))
print("test-set score: {:.3f}".format(grid.score(X_test, y_test)))


# Create a DataFrame from the cv_results_
results_df = pd.DataFrame(grid.cv_results_)
pivot_table = pd.pivot_table(results_df, values='mean_test_score', index='param_n_estimators', columns='param_max_features')

#pivot_table = pd.pivot_table(results_df, values='mean_test_score', index='param_n_estimators', columns='max_features')
sns.heatmap(pivot_table, annot=True, cmap='YlGnBu')


# Add labels and title
plt.xlabel('Max Features')
plt.ylabel('Number of Trees')
plt.title('Random Forest Grid Search Results')
plt.show()



