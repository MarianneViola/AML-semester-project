import numpy as np
from pandas import read_csv
import pandas_profiling as pp

# loader data ind via pandas
filename = '../heart_failure_clinical_records_dataset.csv'
data = read_csv(filename)

# genererer rapport af dataset via pandas_profiling
profile = pp.ProfileReport(data)
profile.to_file("output_heart.html")


#
