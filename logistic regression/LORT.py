from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score
from mlxtend.plotting import plot_confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import RobustScaler
import seaborn as sns
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import numpy as np

# Import data
filename = "heart_failure_clinical_records_dataset.csv"
data = pd.read_csv(filename)

# split the dataset into training and testing subsets
train_df, test_df = train_test_split(data, test_size=0.2, random_state=3)

# Create test and train data
X_test, y_test = test_df.drop(['DEATH_EVENT'], axis=1), test_df['DEATH_EVENT']
X_train, y_train = train_df.drop(['DEATH_EVENT'], axis=1), train_df['DEATH_EVENT']

# undersampling
rus = RandomUnderSampler(replacement=False)
X_train, y_train = rus.fit_resample(X_train, y_train)

############################### logistic regression #####################
X_train_lr = X_train.drop(['serum_creatinine'], axis=1)
X_test_lr = X_test.drop(['serum_creatinine'], axis=1)

lr = Pipeline([
    ('scaler', RobustScaler()),
    ('LR', LogisticRegression(C = 0.1, penalty = 'l2', solver='newton-cg', max_iter=1000, multi_class='ovr', intercept_scaling=10.0, fit_intercept=True, dual=False, class_weight='balanced'))
])
lr.fit(X_train_lr, y_train)

y_proba = lr.predict_proba(X_test_lr)[:, 1]

# Calculate the false positive rate, true positive rate, and thresholds using the ROC curve function
fpr_lr, tpr_lr,_ = roc_curve(y_test, y_proba)

# Calculate the area under the ROC curve
roc_auc_lr = auc(fpr_lr, tpr_lr)

# Make parameter dictionary for the grid search
param_grid_lr = {'LR__C': [0.1, 1, 10, 100, 1000]}

grid = GridSearchCV(lr, param_grid_lr, cv=5, scoring='roc_auc', return_train_score=True, n_jobs=-1)

grid.fit(X_train_lr, y_train)
results = pd.DataFrame(grid.cv_results_)

results.plot('param_LR__C', 'mean_train_score')
results.plot('param_LR__C', 'mean_test_score', ax=plt.gca())
plt.fill_between(results.param_LR__C.astype(np.float64), 
                 results['mean_train_score'] - results['std_train_score'], 
                 results['mean_train_score'] + results['std_train_score'], alpha=0.2)
plt.fill_between(results.param_LR__C.astype(np.float64),
                 results['mean_test_score'] - results['std_test_score'],
                 results['mean_test_score'] + results['std_test_score'], alpha=0.2)
plt.legend(['Train Score', 'Cross-Validation Score'])
plt.show()
