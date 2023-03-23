from matplotlib import pyplot
import pandas as pd

filename = '../heart_failure_clinical_records_dataset.csv'
data = pd.read_csv(filename)

data.plot(kind='box', subplots=True, layout=(4,4), sharex=False, sharey=False, figsize=(15,10))
pyplot.show()
pyplot.savefig('boxplot.png')