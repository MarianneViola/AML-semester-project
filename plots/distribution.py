import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

filename = '../heart_failure_clinical_records_dataset.csv'
data = pd.read_csv(filename)

#plt.rcParams["figure.figsize"] = [7.5, 7.5]
plt.rcParams['figure.constrained_layout.use'] = True
Y = data['DEATH_EVENT']
X = data.drop(['DEATH_EVENT'],axis = 1)
plt.boxplot(X)
plt.xticks(np.arange(1, X.shape[1] + 1), X.columns, rotation=45, ha="right", fontsize=12)
plt.ylabel("DEATH EVENTS", fontsize=12)
plt.yticks(fontsize=12)
plt.savefig('boxplot.png')
plt.show()



