import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
import matplotlib.pyplot as plt

# Load the dataset
# Import data
filename = r"C:\Users\flyve\PycharmProjects\AML_shared\heart_failure_clinical_records_dataset.csv"
heart_df = pd.read_csv(filename)


# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    heart_df.drop('DEATH_EVENT', axis=1), heart_df['DEATH_EVENT'],
    test_size=0.2, random_state=3
)

# Set up the hyperparameter search space
param_grid = {
    'penalty': ['l1', 'l2'],
    'C': [0.1, 1.0, 10.0],
    'solver': ['liblinear', 'saga', 'lbfgs'],
    'max_iter': [100, 500, 1000],
}

# Create a logistic regression model
log_reg = LogisticRegression()

# Set up the grid search with cross-validation
grid = GridSearchCV(
    log_reg, param_grid=param_grid, cv=5, n_jobs=-1, scoring='accuracy'
)

# Fit the grid search to the training data
grid.fit(X_train, y_train)

# Create a DataFrame from the grid search results
results_df = pd.DataFrame(grid.cv_results_)

# Pivot table to show the mean test score for each combination of penalty and C
pivot_table = pd.pivot_table(results_df, values='mean_test_score', index='param_penalty', columns='param_C')

# Create the heatmap
sns.heatmap(pivot_table, annot=True, cmap='YlGnBu')

# Add labels and title
plt.xlabel('C')
plt.ylabel('Penalty')
plt.title('Logistic Regression Grid Search')
plt.show()