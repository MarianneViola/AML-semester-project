from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score
from mlxtend.plotting import plot_confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.pipeline import Pipeline

# Import data
filename = r"C:\Users\flyve\PycharmProjects\AML_shared\heart_failure_clinical_records_dataset.csv"
data = pd.read_csv(filename)


# split the dataset into training and testing subsets
train_df, test_df = train_test_split(data, test_size=0.2, random_state=3)


# Create test and train data
X_test,y_test = test_df.drop(['DEATH_EVENT'], axis=1), test_df['DEATH_EVENT']
X_train,y_train = train_df.drop(['DEATH_EVENT'], axis=1), train_df['DEATH_EVENT']


################################# random forests #########################
# make random forrest classifier
rfc = RandomForestClassifier(random_state=3,n_estimators=100, max_features=3)
rfc.fit(X_train, y_train)

# Predict the probabilities of the test set using the random forest classifier
y_proba = rfc.predict_proba(X_test)[:, 1]

# Calculate the false positive rate, true positive rate, and thresholds using the ROC curve function
fpr_rfc, tpr_rfc, _ = roc_curve(y_test, y_proba)

# Calculate the area under the ROC curve
roc_auc_rfc = auc(fpr_rfc, tpr_rfc)

############################### svm ##################################
# Scale the data using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svm = SVC(kernel='linear', probability=True, C=10, gamma = 0.01, degree = 2, coef0 = 0)
svm.fit(X_train_scaled, y_train)

# Predict the probabilities of the test set using the random forest classifier
y_proba = svm.predict_proba(X_test_scaled)[:, 1]

# Calculate the false positive rate, true positive rate, and thresholds using the ROC curve function
fpr_svm, tpr_svm, _ = roc_curve(y_test, y_proba)

# Calculate the area under the ROC curve
roc_auc_svm = auc(fpr_svm, tpr_svm)

############################### logistic regression #####################
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = LogisticRegression(C = 0.1, penalty = 'l2')
lr.fit(X_train_scaled, y_train)

# Predict the probabilities of the test set using the random forest classifier
y_proba = lr.predict_proba(X_test_scaled)[:, 1]

# Calculate the false positive rate, true positive rate, and thresholds using the ROC curve function
fpr_lr, tpr_lr,_ = roc_curve(y_test, y_proba)

# Calculate the area under the ROC curve
roc_auc_lr = auc(fpr_lr, tpr_lr)


############### Plot the ROC curve ##########################
plt.plot(fpr_rfc, tpr_rfc, color='darkorange', label='Random forrest (area = %0.2f)' % roc_auc_rfc)
plt.plot(fpr_svm, tpr_svm, color='green', label='SVM (area = %0.2f)' % roc_auc_svm)
plt.plot(fpr_lr, tpr_lr, color='red', label='linear regression (area = %0.2f)' % roc_auc_lr)
plt.plot([0, 1], [0, 1], color='navy',linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic (ROC) curve')
plt.legend(loc="lower right")


##################### confusion matrix ################################
y_pred_train_svm = svm.predict(X_train_scaled)
y_pred_train_lr = lr.predict(X_train_scaled)
y_pred_train_rfc = rfc.predict(X_train)

cm_svm = confusion_matrix(y_train,y_pred_train_svm)
cm_lr = confusion_matrix(y_train,y_pred_train_lr)
cm_rfc = confusion_matrix(y_train,y_pred_train_rfc)
fig_svm_train, ax = plot_confusion_matrix(conf_mat=cm_svm, cmap=plt.cm.Blues)
fig_lr_train, ax = plot_confusion_matrix(conf_mat=cm_lr, cmap=plt.cm.Blues)
fig_rfc_train, ax = plot_confusion_matrix(conf_mat=cm_rfc, cmap=plt.cm.Blues)


# find predicted data for the confusion matrix
y_pred_test_svm = svm.predict(X_test_scaled)
y_pred_test_lr = lr.predict(X_test_scaled)
y_pred_test_rfc = rfc.predict(X_test)


# Compute confusion matrices
cms = []
cms.append(confusion_matrix(y_test,y_pred_test_svm))
cms.append(confusion_matrix(y_test,y_pred_test_lr))
cms.append(confusion_matrix(y_test,y_pred_test_rfc))

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
print("SVM accuracy score {:.3f}".format(accuracy_score(y_test, y_pred_test_svm)))
print("LR accuracy score {:.3f}".format(accuracy_score(y_test, y_pred_test_lr)))
print("RF accuracy score {:.3f}".format(accuracy_score(y_test, y_pred_test_rfc)))
print("")

#################### Calculate sensitivity and specificity #######################
##svm
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_test_svm).ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
print(tn,tp,fn,fp)
print("SVM sensitivity {}".format(sensitivity))
print("SVM specificity {}".format(specificity))

## lr
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_test_lr).ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
print("lr sensitivity {}".format(sensitivity))
print("lr specificity {}".format(specificity))

## rfc
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_test_rfc).ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
print("rfc sensitivity {}".format(sensitivity))
print("rfc specificity {}".format(specificity))


#################### Difference between genders ##########################
# filter the dataframe into two different genders
male_train_df = train_df.loc[train_df['sex'] == 1]
female_train_df = train_df.loc[train_df['sex'] == 0]

male_test_df = test_df.loc[test_df['sex'] == 1]
female_test_df = test_df.loc[test_df['sex'] == 0]

# make test and train datasets
X_female_train, y_female_train =  scaler.transform(female_train_df.drop(['DEATH_EVENT'], axis=1)), female_train_df['DEATH_EVENT']
X_female_test, y_female_test = scaler.transform(female_test_df.drop(['DEATH_EVENT'], axis=1)), female_test_df['DEATH_EVENT']
X_male_train, y_male_train = scaler.transform(male_train_df.drop(['DEATH_EVENT'], axis=1)), male_train_df['DEATH_EVENT']
X_male_test, y_male_test = scaler.transform(male_test_df.drop(['DEATH_EVENT'], axis=1)), male_test_df['DEATH_EVENT']

# scale_it
X_female_train_s = scaler.transform(X_female_train)
X_female_test_s = scaler.transform(X_female_test)
X_male_train_s = scaler.transform(X_male_train)
X_male_test_s = scaler.transform(X_male_test)

print("SVM Male test score {:.3f}".format(accuracy_score(y_male_test, svm.predict(X_male_test_s))))
print("SVM Female test score {:.3f}".format(accuracy_score(y_female_test, svm.predict(X_female_test_s))))
print("LR Male test score {:.3f}".format(accuracy_score(y_male_test, lr.predict(X_male_test_s))))
print("LR Female test score {:.3f}".format(accuracy_score(y_female_test, lr.predict(X_female_test_s))))
print("RFC Male test score {:.3f}".format(accuracy_score(y_male_test, rfc.predict(X_male_test))))
print("RFC Female test score {:.3f}".format(accuracy_score(y_female_test,rfc.predict(X_female_test))))
print("")

