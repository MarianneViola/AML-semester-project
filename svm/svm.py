import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score
#from sklearn.metrics import plot_roc_curve 
from sklearn.preprocessing import StandardScaler
import numpy as np



# Load the dataset
heart_data = pd.read_csv('heart_failure_clinical_records_dataset.csv')

array = heart_data.values
X = array[:,0:8]
y = array[:,8]
import seaborn as sns
import matplotlib.pyplot as plt


correlation = heart_data.corr()

plt.figure(figsize=(10,6))
sns.heatmap(correlation, annot=True)


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(heart_data.iloc[:, :-1], heart_data.iloc[:, -1], test_size=0.3, random_state=3)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create an SVM pipeline with an RBF kernel
svm_pipeline = SVC(kernel='rbf', probability=True)

# Define the parameter grid for GridSearchCV
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [0.01, 0.1, 1, 10], 'coef0': [0, 1, 2], 'degree': [2, 3, 4], 'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}

# Perform GridSearchCV to find the best hyperparameters
svm_grid = GridSearchCV(svm_pipeline, param_grid, cv=5, return_train_score=True)
svm_grid.fit(X_train, y_train)

# Print the best hyperparameters
print("Best hyperparameters:", svm_grid.best_params_)

# Use the best hyperparameters to fit the SVM pipeline
svm_pipeline.set_params(**svm_grid.best_params_)
svm_pipeline.fit(X_train, y_train)

# Calculate the test and train scores
train_score = svm_pipeline.score(X_train, y_train)
test_score = svm_pipeline.score(X_test, y_test)

# Print the test and train scores
print("Train score:", train_score)
print("Test score:", test_score)

# Make predictions on the test set and print the classification report
y_pred = svm_pipeline.predict(X_test)
print(classification_report(y_test, y_pred, zero_division=1))

# Plot the ROC curve
#plot_roc_curve(svm_pipeline, X_test, y_test)

# Calculate the AUC
auc = roc_auc_score(y_test, svm_pipeline.predict_proba(X_test)[:, 1])
print("AUC:", auc)

from sklearn.metrics import accuracy_score

# Make predictions on the test set
y_pred = svm_pipeline.predict(X_test)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)

# Print the accuracy
print("Accuracy:", accuracy)






