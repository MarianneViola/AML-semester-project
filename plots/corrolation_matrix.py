import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
filename = r"C:\Users\flyve\PycharmProjects\AML_shared\heart_failure_clinical_records_dataset.csv"
heart_data = pd.read_csv(filename)

array = heart_data.values
X = array[:,0:8]
y = array[:,8]

correlation = heart_data.corr().round(2)

plt.rcParams['figure.constrained_layout.use'] = True
plt.figure(figsize=(10,6))
sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0, vmin=-1, vmax=1, annot_kws={'fontsize':10, 'fontweight':'bold'})
plt.show()