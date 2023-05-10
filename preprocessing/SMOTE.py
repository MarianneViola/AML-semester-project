import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import keras
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler

# Load the dataset
heart_data = pd.read_csv('../heart_failure_clinical_records_dataset.csv')

#Create an instance of the SMOTE class
sm = SMOTE(random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(heart_data.iloc[:, :-1], heart_data.iloc[:, -1], test_size=0.3)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


#Fit and resample the training data using SMOTE
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)


# Define the model
model = keras.Sequential()
model.add(Dense(20, input_dim=X_train.shape[1], activation='relu', kernel_initializer='uniform'))
model.add(Dense(10, activation='relu', kernel_initializer='uniform'))
model.add(Dense(1, activation='sigmoid', kernel_initializer='uniform'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#Train and evaluate the machine learning model using the resampled training data (X_train_res and y_train_res):
model.fit(X_train_res, y_train_res)
score_tr = model.evaluate(X_train_res, y_train_res)
score_ts = model.evaluate(X_test, y_test)