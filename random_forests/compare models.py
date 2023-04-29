import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

filename = r"C:\Users\flyve\PycharmProjects\AML_shared\heart_failure_clinical_records_dataset.csv"
data = pd.read_csv(filename)

# create x and y and make test and train data
x,y = data.drop(['DEATH_EVENT'], axis=1), data['DEATH_EVENT']
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=3)

# make random forrest classifier
rfc1 = RandomForestClassifier(random_state=3,n_estimators=100, max_features=3, oob_score = True)
rfc1.fit(X_train, y_train)
print("rfc1 oob_score ", rfc1.oob_score_)
print(rfc1.score(X_test, y_test))

rfc1 = RandomForestClassifier(max_depth = 44,max_leaf_nodes = 35, min_impurity_decrease= 0.03, min_samples_split=3, random_state=3,n_estimators=45, max_features='auto', oob_score = True)
rfc1.fit(X_test, y_test)
print("rfc2 oob_score ", rfc1.oob_score_)
print(rfc1.score(X_train, y_train))





