import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Import data
filename = r"/heart_failure_clinical_records_dataset.csv"
data = pd.read_csv(filename)

# Create test and train data
x,y = data.drop(['DEATH_EVENT'], axis=1), data['DEATH_EVENT']
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=3)

# make random forrest classifier
rfc = RandomForestClassifier(random_state=3)
rfc.fit(x, y)

# Plot feature importances through Gini impurity
importances = rfc.feature_importances_
sorted_idx = np.argsort(importances)

plt.rcParams['figure.constrained_layout.use'] = True
plt.barh(range(data.shape[1]-1), importances[sorted_idx], align='center')
plt.yticks(range(data.shape[1]-1), data.columns[sorted_idx])
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Random Forests feature selection')
plt.show()

import pandas as pd
from scipy.stats import chi2_contingency


# specify the target feature
target_feature = 'DEATH_EVENT'

# create an empty DataFrame to store the p-values
p_values_df = pd.DataFrame(index=data.columns[:-1], columns=["p-value"])

# loop through each feature and calculate the chi-squared p-value
for feature in data.columns[:-1]:
    contingency_table = pd.crosstab(data[feature], data[target_feature])
    _, p_value, _, _ = chi2_contingency(contingency_table)
    p_values_df.loc[feature, "p-value"] = p_value

# print the p-values
print(p_values_df)