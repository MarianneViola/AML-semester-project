import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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
plt.savefig('boxplot_seperated.png')
plt.show()