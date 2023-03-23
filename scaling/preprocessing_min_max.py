import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np

# loader data ind via pandas
filename = '../heart_failure_clinical_records_dataset.csv'
data = pd.read_csv(filename)

# find x og y til classificerings KNN
x,y = data.drop(['DEATH_EVENT'], axis=1), data['DEATH_EVENT']
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)

# scaler data (find fit) for training data
scaler_MinMax = MinMaxScaler()
scaler_MinMax.fit(X_train)
X_train_scaled = scaler_MinMax.transform(X_train)

# beregn hvor god accuracy vi har med og uden scalering
knn_scaled = KNeighborsClassifier().fit(X_train_scaled, y_train)
X_test_scaled = scaler_MinMax.transform(X_test)
print('Scaled: {}'.format(knn_scaled.score(X_test_scaled, y_test)))

knn_unscaled = KNeighborsClassifier().fit(X_train, y_train)
print('Not scaled: {}'.format(knn_unscaled.score(X_test, y_test)))

# kig på mean og sd
sd = np.std(X_train_scaled[:,0])
mean = np.mean(X_train_scaled[:,0])
print(sd, mean)