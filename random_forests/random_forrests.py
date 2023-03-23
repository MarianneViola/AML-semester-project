from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from random_forests.plot_accuracy import plot_acc

filename = '../heart_failure_clinical_records_dataset.csv'
data = pd.read_csv(filename)

# create x and y and make test and train data
x,y = data.drop(['DEATH_EVENT'], axis=1), data['DEATH_EVENT']
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=3)

train_scores = []
test_scores = []
oob_scores = []

feature_range = range(1, 15, 1)
for max_features in feature_range:
    rf = RandomForestClassifier(max_features=max_features, oob_score=True, n_estimators=800, random_state=0)
    rf.fit(X_train, y_train)
    train_scores.append(rf.score(X_train, y_train))
    test_scores.append(rf.score(X_test, y_test))
    oob_scores.append(rf.oob_score_)

plot_acc(feature_range, test_scores,oob_scores, train_scores)

train_scores = []
test_scores = []
oob_scores = []
feature_range = range(100, 1000, 100)
for max_features in feature_range:
    rf = RandomForestClassifier(max_features=2, oob_score=True, n_estimators=max_features, random_state=3)
    rf.fit(X_train, y_train)
    train_scores.append(rf.score(X_train, y_train))
    test_scores.append(rf.score(X_test, y_test))
    oob_scores.append(rf.oob_score_)

plot_acc(feature_range, test_scores, oob_scores, train_scores)