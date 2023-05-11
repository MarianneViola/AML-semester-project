from sklearn.ensemble import RandomForestClassifier, VotingClassifier
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
from imblearn.under_sampling import RandomUnderSampler

# Import data
filename = r"C:\Users\flyve\PycharmProjects\AML_shared\heart_failure_clinical_records_dataset.csv"
data = pd.read_csv(filename)
#data = data.drop(['anaemia','smoking','diabetes','high_blood_pressure', 'sex'], axis = 1)

# split the dataset into training and testing subsets
train_df, test_df = train_test_split(data, test_size=0.2, random_state=3)

# Create test and train data
X_test, y_test = test_df.drop(['DEATH_EVENT'], axis=1), test_df['DEATH_EVENT']
X_train, y_train = train_df.drop(['DEATH_EVENT'], axis=1), train_df['DEATH_EVENT']

# SMOTE
#sm = SMOTE(random_state=42)
#X_train, y_train = sm.fit_resample(X_train, y_train)

# undersampling
rus = RandomUnderSampler(replacement=False)
X_train, y_train = rus.fit_resample(X_train, y_train)

################################# random forests #########################
# select the features
X, y = data.drop(['DEATH_EVENT','sex'], axis=1), data['DEATH_EVENT']
X_train_rfc, X_test_rfc, y_train_rfc, y_test_rfc = train_test_split(X, y, random_state=3)

# make the model
rfc = RandomForestClassifier(random_state=3,n_estimators=150, max_leaf_nodes=8,max_depth=18, max_features=10)
rfc.fit(X_train_rfc, y_train_rfc)

# Predict the probabilities of the test set using the random forest classifier
y_proba = rfc.predict_proba(X_test_rfc)[:, 1]

# Calculate the false positive rate, true positive rate, and thresholds using the ROC curve function
fpr_rfc, tpr_rfc, _ = roc_curve(y_test_rfc, y_proba)

# Calculate the area under the ROC curve
roc_auc_rfc = auc(fpr_rfc, tpr_rfc)

############################### svm ##################################
# Scale the data using StandardScaler
svm = Pipeline([
    ('scaler', RobustScaler()),
    ('SVM', SVC(kernel='poly', probability=True, C=100, gamma = 0.01, degree = 2, coef0 = 2))
])
svm.fit(X_train, y_train)

# Predict the probabilities of the test set using the random forest classifier
y_proba = svm.predict_proba(X_test)[:, 1]

# Calculate the false positive rate, true positive rate, and thresholds using the ROC curve function
fpr_svm, tpr_svm, _ = roc_curve(y_test, y_proba)

# Calculate the area under the ROC curve
roc_auc_svm = auc(fpr_svm, tpr_svm)

############################### logistic regression #####################

lr = Pipeline([
    ('scaler', RobustScaler()),
    ('LR', LogisticRegression(C = 1, penalty = 'l2', solver='liblinear', max_iter=100))
])
lr.fit(X_train, y_train)

y_proba = lr.predict_proba(X_test)[:, 1]
# Calculate the false positive rate, true positive rate, and thresholds using the ROC curve function
fpr_lr, tpr_lr,_ = roc_curve(y_test, y_proba)

# Calculate the area under the ROC curve
roc_auc_lr = auc(fpr_lr, tpr_lr)


############## collecting models ##############################
col = VotingClassifier(estimators=[('lr', lr), ('rf', rfc), ('svm', svm)], voting='soft')
col = col.fit(X_train, y_train)


y_proba = col.predict_proba(X_test)[:, 1]
# Calculate the false positive rate, true positive rate, and thresholds using the ROC curve function
fpr_col, tpr_col,_ = roc_curve(y_test, y_proba)

# Calculate the area under the ROC curve
roc_auc_col = auc(fpr_col, tpr_col)

############### Plot the ROC curve ##########################
plt.plot(fpr_rfc, tpr_rfc, color='darkorange', label='Random forrest (area = %0.2f)' % roc_auc_rfc)
plt.plot(fpr_svm, tpr_svm, color='green', label='SVM (area = %0.2f)' % roc_auc_svm)
plt.plot(fpr_lr, tpr_lr, color='red', label='logistic regression (area = %0.2f)' % roc_auc_lr)
plt.plot(fpr_col, tpr_col, color='purple', label='Collected model (area = %0.2f)' % roc_auc_col)
plt.plot([0, 1], [0, 1], color='navy',linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic (ROC) curve')
plt.legend(loc="lower right")


##################### confusion matrix ################################

y_pred_train_svm = svm.predict(X_train)
y_pred_train_lr = lr.predict(X_train)
y_pred_train_rfc = rfc.predict(X_train_rfc)
y_pred_train_col = col.predict(X_train)

cm_svm = confusion_matrix(y_train,y_pred_train_svm)
cm_lr = confusion_matrix(y_train,y_pred_train_lr)
cm_rfc = confusion_matrix(y_train_rfc,y_pred_train_rfc)

fig_svm_train, ax = plot_confusion_matrix(conf_mat=cm_svm, cmap=plt.cm.Blues)
fig_lr_train, ax = plot_confusion_matrix(conf_mat=cm_lr, cmap=plt.cm.Blues)
fig_rfc_train, ax = plot_confusion_matrix(conf_mat=cm_rfc, cmap=plt.cm.Blues)


# find predicted data for the confusion matrix
X_test_scaled = svm.named_steps['scaler'].transform(X_test)
y_pred_test_svm = svm.predict(X_test)
y_pred_test_lr = lr.predict(X_test)
y_pred_test_rfc = rfc.predict(X_test_rfc)
y_pred_test_col = col.predict(X_test)


# Compute confusion matrices
cms = []
cms.append(confusion_matrix(y_test,y_pred_test_svm))
cms.append(confusion_matrix(y_test,y_pred_test_lr))
cms.append(confusion_matrix(y_test_rfc,y_pred_test_rfc))

# Define label names
labels = ['Negative', 'Positive']

# Define plot titles
titles = ['Confusion Matrix (Counts)', 'Confusion Matrix (Normalized by Row)', 'Confusion Matrix (Normalized by Column)']

# Plot confusion matrices as heatmaps
for i, cm in enumerate(cms):
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title(titles[i])
    plt.show()

#################### Calculate accuracy ##############################
print("SVM accuracy score {:.3f}".format(svm.score(X_test,y_test)))
print("LR accuracy score {:.3f}".format(accuracy_score(y_test, y_pred_test_lr)))
print("RF accuracy score {:.3f}".format(accuracy_score(y_test_rfc, y_pred_test_rfc)))
print("col accuracy score {:.3f}".format(accuracy_score(y_test, y_pred_test_col)))
print("")

print("SVM train accuracy score {:.3f}".format(svm.score(X_train,y_train)))
print("LR train accuracy score {:.3f}".format(accuracy_score(y_train, y_pred_train_lr)))
print("RF train accuracy score {:.3f}".format(accuracy_score(y_train_rfc, y_pred_train_rfc)))
print("col train accuracy score {:.3f}".format(accuracy_score(y_train, y_pred_train_col)))
print("")

#################### Calculate sensitivity and specificity #######################
print("--------------test-----------------------")
##svm
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_test_svm).ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
print("SVM sensitivity {:.3f}".format(sensitivity))
print("SVM specificity {:.3f}".format(specificity))

## lr
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_test_lr).ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
print("lr sensitivity {:.3f}".format(sensitivity))
print("lr specificity {:.3f}".format(specificity))

## rfc
tn, fp, fn, tp = confusion_matrix(y_test_rfc, y_pred_test_rfc).ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
print("rfc sensitivity {:.3f}".format(sensitivity))
print("rfc specificity {:.3f}".format(specificity))

## collected
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_test_col).ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
print("Col sensitivity {:.3f}".format(sensitivity))
print("Col specificity {:.3f}".format(specificity))

print("------------------train--------------------------")
tn, fp, fn, tp = confusion_matrix(y_train, y_pred_train_svm).ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
print("SVM sensitivity {:.3f}".format(sensitivity))
print("SVM specificity {:.3f}".format(specificity))

## lr
tn, fp, fn, tp = confusion_matrix(y_train, y_pred_train_lr).ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
print("lr sensitivity {:.3f}".format(sensitivity))
print("lr specificity {:.3f}".format(specificity))

## rfc
tn, fp, fn, tp = confusion_matrix(y_train_rfc, y_pred_train_rfc).ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
print("rfc sensitivity {:.3f}".format(sensitivity))
print("rfc specificity {:.3f}".format(specificity))

## collected
tn, fp, fn, tp = confusion_matrix(y_train, y_pred_train_col).ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
print("col sensitivity {:.3f}".format(sensitivity))
print("col specificity {:.3f}".format(specificity))
print("")

#################### Difference between genders ##########################
# filter the dataframe into two different genders
male_train_df = train_df.loc[train_df['sex'] == 1]
female_train_df = train_df.loc[train_df['sex'] == 0]

male_test_df = test_df.loc[test_df['sex'] == 1]
female_test_df = test_df.loc[test_df['sex'] == 0]

# make test and train datasets
X_female_train, y_female_train = female_train_df.drop(['DEATH_EVENT'], axis=1), female_train_df['DEATH_EVENT']
X_female_test, y_female_test = female_test_df.drop(['DEATH_EVENT'], axis=1), female_test_df['DEATH_EVENT']
X_male_train, y_male_train = male_train_df.drop(['DEATH_EVENT'], axis=1), male_train_df['DEATH_EVENT']
X_male_test, y_male_test = male_test_df.drop(['DEATH_EVENT'], axis=1), male_test_df['DEATH_EVENT']

y_pred_test_svm = svm.predict(X_test_scaled)

print("SVM Male test score {:.3f}".format(accuracy_score(y_male_test, svm.predict(X_male_test))))
print("SVM Female test score {:.3f}".format(accuracy_score(y_female_test, svm.predict(X_female_test))))
print("LR Male test score {:.3f}".format(accuracy_score(y_male_test, lr.predict(X_male_test))))
print("LR Female test score {:.3f}".format(accuracy_score(y_female_test, lr.predict(X_female_test))))
print("RFC Male test score {:.3f}".format(accuracy_score(y_male_test, rfc.predict(X_male_test))))
print("RFC Female test score {:.3f}".format(accuracy_score(y_female_test,rfc.predict(X_female_test))))
print("")

