import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

filename = '../heart_failure_clinical_records_dataset.csv'
data = pd.read_csv(filename)

# Heatmap of the whole data
#plt.figure(figsize = (15,8))
plt.rcParams['figure.constrained_layout.use'] = True
heatmap = sns.heatmap(data.corr(),annot=False, cbar=True, cmap='coolwarm')
plt.show()



