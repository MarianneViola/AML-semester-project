import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import RobustScaler

filename = '../heart_failure_clinical_records_dataset.csv'
df = pd.read_csv(filename)

# set style
sns.set_style("whitegrid")

# create boxplot for each variable in a separate subplot
fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(16, 10))

for ax, col in zip(axes.flatten(), df.columns[:-2]):
    sns.boxplot(x='DEATH_EVENT', y=col, data=df, ax=ax)
    ax.set_title(f'Boxplot of {col}')

# plot the last variable 'time'
sns.boxplot(x='DEATH_EVENT', y='time', data=df, ax=axes.flatten()[-1])
axes.flatten()[-1].set_title('Boxplot of time')

plt.tight_layout()
plt.savefig('boxplot_separated.png')
plt.show()