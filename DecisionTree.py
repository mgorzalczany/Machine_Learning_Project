from clean_up import Classifiers, dane
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

c=Classifiers()
X_clean = c.datasetPreprocessing(
     X = dane,columns_to_map = ['workclass','education','age','marital-status','occupation',
                                'relationship', 'race', 'sex', 'native-country', 'y'])
print(X_clean.dtypes)

X_train, X_test, y_train, y_test = c.splitDatasetIntoTrainAndTest(
      X=X_clean.drop(columns=['y']),
      y=X_clean['y'])

dt = DecisionTreeClassifier()
dt.fit(X_train,y_train)

y_pred = dt.predict(X_test)

acc_score = accuracy_score(y_test, y_pred)
con_matrix = confusion_matrix(y_test, y_pred)

print(acc_score)
print(con_matrix)
