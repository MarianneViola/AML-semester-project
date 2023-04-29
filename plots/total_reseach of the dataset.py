import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

######################## Corrolation Matrix ################################

######################## Multiple different boxplots #######################
filename = '../heart_failure_clinical_records_dataset.csv'
df = pd.read_csv(filename)
print(df.columns)

# set style
sns.set_style("whitegrid")

# create boxplot for each variable in a separate subplot
fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(16, 10))

for ax, col in zip(axes.flatten(), df.columns[:-2]):
    boxplot = df.boxplot(column=col, ax=ax)
    ax.set_title(f'Boxplot of {col}')

# plot the last variable 'time'
boxplot = df.boxplot(column='time', ax=axes.flatten()[-1])
axes.flatten()[-1].set_title('Boxplot of time')

plt.tight_layout()

######################## All features in one boxplot #######################

plt.rcParams['figure.constrained_layout.use'] = True
Y = df['DEATH_EVENT']
X = df.drop(['DEATH_EVENT'],axis = 1)
plt.boxplot(X)
plt.xticks(np.arange(1, X.shape[1] + 1), X.columns, rotation=45, ha="right", fontsize=12)
plt.ylabel("DEATH EVENTS", fontsize=12)
plt.yticks(fontsize=12)
plt.savefig('boxplot.png')
plt.show()

######################## Data Balance #######################
counts = df['DEATH_EVENT'].value_counts()
counts_percentage = df['DEATH_EVENT'].value_counts(normalize=True) * 100
print(counts)
print(counts_percentage)