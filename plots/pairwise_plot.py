import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

filename = '../heart_failure_clinical_records_dataset.csv'
data = pd.read_csv(filename)
df_sorted = data.drop(['anaemia','diabetes','high_blood_pressure', 'sex','smoking'], axis = 1)



# alle mulige plots
sns.set_style("whitegrid")

cols = df_sorted.columns[:-1].tolist()

g = sns.FacetGrid(pd.DataFrame(cols), col=0, col_wrap=3, sharex=False)

for ax, varx in zip(g.axes, cols):
    sns.scatterplot(data=df_sorted, x=varx, y="DEATH_EVENT", ax=ax)

g.tight_layout()

#sns.pairplot(data, y_vars=["DEATH_EVENT"],x_vars = data.columns[:-1])
print(2)
plt.show()
plt.savefig('pairwise_plot.png')