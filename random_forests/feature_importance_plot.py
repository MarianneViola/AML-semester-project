import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Import data
filename = r"C:\Users\flyve\PycharmProjects\AML_shared\heart_failure_clinical_records_dataset.csv"
data = pd.read_csv(filename)

# Create test and train data
x,y = data.drop(['DEATH_EVENT'], axis=1), data['DEATH_EVENT']
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=3)

# make random forrest classifier
rfc = RandomForestClassifier(random_state=3, n_estimators=100, max_features=3)
rfc.fit(X_train, y_train)

plt.barh(range(data.shape[1]-1), np.sort(rfc.feature_importances_[0:data.shape[1]-1]))
plt.yticks(range(data.shape[1]-1), data.columns.tolist()[0:data.shape[1]-1])
plt.tick_params(axis='x', labelsize=20)
plt.tick_params(axis='y', labelsize=20)
plt.show()