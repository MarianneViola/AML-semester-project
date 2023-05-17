from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score
from mlxtend.plotting import plot_confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import RobustScaler
import seaborn as sns
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from mlxtend.feature_selection import SequentialFeatureSelector

# Import data
filename = "heart_failure_clinical_records_dataset.csv"
data = pd.read_csv(filename)

# split the dataset into training and testing subsets
train_df, test_df = train_test_split(data, test_size=0.2, random_state=3)

# Create test and train data
X_test,y_test = test_df.drop(['DEATH_EVENT'], axis=1), test_df['DEATH_EVENT']
X_train,y_train = train_df.drop(['DEATH_EVENT'], axis=1), train_df['DEATH_EVENT']

# define your models
lr = LogisticRegression(C = 100, penalty = 'l2', solver='liblinear', max_iter=100)
rfc = RandomForestClassifier(random_state=3,n_estimators=135, max_leaf_nodes=14,max_depth=6)
svm = Pipeline([
    ('scaler', RobustScaler()),
    ('SVM', SVC(kernel='linear', probability=True, C=10, gamma = 0.01, degree = 2, coef0 = 0))
])

# define the feature selector
bfsLR = SequentialFeatureSelector(lr, k_features='best', cv=5, forward=False, n_jobs=-1)
bfsRFC = SequentialFeatureSelector(rfc, k_features='best', cv=5, forward=False, n_jobs=-1)
bfsSVM = SequentialFeatureSelector(svm, k_features='best', cv=5, forward=False, n_jobs=-1)

# fit the feature selector on your training data
bfsLR = bfsLR.fit(X_train, y_train)
bfsRFC = bfsRFC.fit(X_train, y_train)
bfsSVM = bfsSVM.fit(X_train, y_train)

# get the selected features
selected_featuresLR = X_train.columns[list(bfsLR.k_feature_idx_)]
selected_featuresRFC = X_train.columns[list(bfsRFC.k_feature_idx_)]
selected_featuresSVM = X_train.columns[list(bfsSVM.k_feature_idx_)]

# print out the excluded features
excluded_featuresLR = X_train.drop(columns=selected_featuresLR).columns.tolist()
excluded_featuresRFC = X_train.drop(columns=selected_featuresRFC).columns.tolist()
excluded_featuresSVM = X_train.drop(columns=selected_featuresSVM).columns.tolist()

print('LR excluded features: ', excluded_featuresLR)
print('RFC excluded features: ', excluded_featuresRFC)
print('SVM excluded features: ', excluded_featuresSVM)

# train your models on the selected features
lr.fit(X_train[selected_featuresLR], y_train)
rfc.fit(X_train[selected_featuresRFC], y_train)
svm.fit(X_train[selected_featuresSVM], y_train)

# select the same features in the test set as in the training set
X_test_LR = X_test[selected_featuresLR]
X_test_RFC = X_test[selected_featuresRFC]
X_test_SVM = X_test[selected_featuresSVM]

# find predicted data for the confusion matrix
X_test_scaled = svm.named_steps['scaler'].transform(X_test_SVM)
y_pred_test_svm = svm.predict(X_test_SVM)
y_pred_test_lr = lr.predict(X_test_LR)
y_pred_test_rfc = rfc.predict(X_test_RFC)


# evaluate the performance of your model
print('LR With feature selection test: {}'.format(lr.score(X_test[selected_featuresLR], y_test)))
print('RFC With feature selection test: {}'.format(rfc.score(X_test[selected_featuresRFC], y_test)))
print('SVM With feature selection test: {}'.format(svm.score(X_test[selected_featuresSVM], y_test)))

#################### Calculate sensitivity and specificity #######################
##svm
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_test_svm).ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
print(tn,tp,fn,fp)
print("SVM sensitivity {:.3f}".format(sensitivity))
print("SVM specificity {:.3f}".format(specificity))

## lr
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_test_lr).ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
print("lr sensitivity {:.3f}".format(sensitivity))
print("lr specificity {:.3f}".format(specificity))

## rfc
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_test_rfc).ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
print("rfc sensitivity {:.3f}".format(sensitivity))
print("rfc specificity {:.3f}".format(specificity))