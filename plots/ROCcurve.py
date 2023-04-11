from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# Import data
filename = r"C:\Users\flyve\PycharmProjects\AML_shared\heart_failure_clinical_records_dataset.csv"
data = pd.read_csv(filename)

# Create test and train data
x,y = data.drop(['DEATH_EVENT'], axis=1), data['DEATH_EVENT']
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=3)

################################# random forests #########################
# make random forrest classifier
rfc = RandomForestClassifier(random_state=3,n_estimators=100, max_features=3)
rfc.fit(X_train, y_train)

# Predict the probabilities of the test set using the random forest classifier
y_proba = rfc.predict_proba(X_test)[:, 1]

# Calculate the false positive rate, true positive rate, and thresholds using the ROC curve function
fpr_rfc, tpr_rfc, thresholds = roc_curve(y_test, y_proba)

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
fpr_svm, tpr_svm, thresholds = roc_curve(y_test, y_proba)

# Calculate the area under the ROC curve
roc_auc_svm = auc(fpr_svm, tpr_svm)


############################### logistic regression #####################
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

log_reg = LogisticRegression(C = 0.1, penalty = 'l2')
log_reg.fit(X_train_scaled, y_train)

# Predict the probabilities of the test set using the random forest classifier
y_proba = log_reg.predict_proba(X_test_scaled)[:, 1]

# Calculate the false positive rate, true positive rate, and thresholds using the ROC curve function
fpr_lr, tpr_lr,_ = roc_curve(y_test, y_proba)

# Calculate the area under the ROC curve
roc_auc_lr = auc(fpr_lr, tpr_lr)


# Plot the ROC curve
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


# find predicted data for the confusion matrix
y_pred_test_svm = svm.predict(X_test_scaled)
y_pred_test_lr = log_reg.predict(X_test_scaled)
y_pred_test_rfc = rfc.predict(X_test)

cm_svm = confusion_matrix(y_pred_test_svm,y_test)
cm_lr = confusion_matrix(y_pred_test_lr,y_test)
cm_rfc = confusion_matrix(y_pred_test_rfc,y_test)
fig_svm, ax = plot_confusion_matrix(conf_mat=cm_svm, cmap=plt.cm.Blues)
fig_lr, ax = plot_confusion_matrix(conf_mat=cm_lr, cmap=plt.cm.Blues)
fig_rfc, ax = plot_confusion_matrix(conf_mat=cm_rfc, cmap=plt.cm.Blues)


# do the plots for the train data
y_pred_train_svm = svm.predict(X_train_scaled)
y_pred_train_lr = log_reg.predict(X_train_scaled)
y_pred_train_rfc = rfc.predict(X_train)

cm_svm = confusion_matrix(y_pred_train_svm,y_train)
cm_lr = confusion_matrix(y_pred_train_lr,y_train)
cm_rfc = confusion_matrix(y_pred_train_rfc,y_train)
fig_svm_train, ax = plot_confusion_matrix(conf_mat=cm_svm, cmap=plt.cm.Blues)
fig_lr_train, ax = plot_confusion_matrix(conf_mat=cm_lr, cmap=plt.cm.Blues)
fig_rfc_train, ax = plot_confusion_matrix(conf_mat=cm_rfc, cmap=plt.cm.Blues)
plt.show()


