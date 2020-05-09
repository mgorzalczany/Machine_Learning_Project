import pandas as pd

dane = pd.read_csv('Adult_train.tab', sep = "\t")

pd.set_option('display.max_columns', None)

print(dane.dtypes)
print(dane.isnull().count())
print(dane.describe())
print(dane.shape)

dane = dane.replace("?", pd.np.nan)
print(dane.isnull().sum())