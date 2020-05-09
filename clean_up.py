import pandas as pd

dane = pd.read_csv('Adult_train.tab', sep = "\t")

pd.set_option('display.max_columns', None)

print(dane.dtypes)
print(dane.isnull().count())
print(dane.describe())
print(dane.shape)

dane = dane.replace("?", pd.np.nan)

dane.dropna(subset=['native-country'])
print(dane.isnull().sum())

# df['DataFrame Column'] = df['DataFrame Column'].fillna(0)

dane['native-country'] = dane['native-country'] .fillna(0)
dane['workclass'] = dane['workclass'] .fillna(0)
dane['occupation'] = dane['occupation'] .fillna(0)
print(dane.isnull().sum())
print(dane.dtypes)