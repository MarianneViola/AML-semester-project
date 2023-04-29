from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
              'logisticregression__penalty': ['l1', 'l2']}

# Create a GridSearchCV object and fit it to the data
grid = GridSearchCV(pipe, param_grid=param_grid, cv=5)
grid.fit(X_train, y_train)

# Print the best hyperparameters and the corresponding mean cross-validation score
print("Best hyperparameters: ", grid.best_params_)
print("Best score: ", grid.best_score_)

# Extract the mean test scores for each hyperparameter combination
mean_test_scores = grid.cv_results_['mean_test_score'].reshape(len(param_grid['logisticregression__C']),
                                                               len(param_grid['logisticregression__penalty']))

# Create a heatmap of the mean test scores
sns.heatmap(mean_test_scores, annot=True, fmt='.3f',
            xticklabels=param_grid['logisticregression__penalty'],
            yticklabels=param_grid['logisticregression__C'],
            cmap='coolwarm')
plt.xlabel('Penalty')
plt.ylabel('C')
plt.title('Mean Test Scores')
plt.show()
