import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

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

#MAPOWANIE#

class Classifiers:
    def datasetPreprocessing(self, X, columns_to_map):
        X_clean=X
        # mapowanie
        for column_name in columns_to_map:
            # konstruowanie mappera
            mapper = {}
            for index, category in enumerate(X_clean[column_name].unique()):
                mapper[category] = index
            # mapowanie
            X_clean[column_name] = X_clean[column_name].map(mapper)

        return X_clean
#   def trainAndTestClassifier(self, clf, X_train, X_test, y_train):

c=Classifiers()
X_clean = c.datasetPreprocessing(
     X = dane,columns_to_map = ['workclass','education','age','marital-status','occupation',
                                'relationship', 'race', 'sex', 'native-country', 'y'])
print(X_clean.dtypes)


#MAPOWANIE#


class Classifiers:
    def datasetPreprocessing(self, X, columns_to_map):
        X_clean=X
        # mapowanie
        for column_name in columns_to_map:
            # konstruowanie mappera
            mapper = {}
            for index, category in enumerate(X_clean[column_name].unique()):
                mapper[category] = index
            # mapowanie
            X_clean[column_name] = X_clean[column_name].map(mapper)

        return X_clean

    def splitDatasetIntoTrainAndTest(self, X, y, train_split_percent=0.6):
        # pd.set_option('display.max_columns', None)
        # print(X)
        print(X.info())
        # print(X.describe())
        # print(X.describe(include=[pd.np.number]))
        # print(X.describe(include=[pd.np.object]))
        # print(X.describe(include=['category']))
        # print(X.describe(include={'boolean'}))
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_split_percent)
        return X_train, X_test, y_train, y_test
    #   def trainAndTestClassifier(self, clf, X_train, X_test, y_train):
    def ensableClassifier(self, clfs, X_train, X_test, y_train):
        y_preds = []
        # trenowanie i testowanie wszystkich klasyfikatorów z listy clfs
        for clf in clfs:
            clf.fit(X_train, y_train)
            y_preds.append(clf.predict(X_test))
        # głosowanie większościowe
        y_result = y_preds[0]
        clf_index = 1
        while (clf_index < len(y_preds)):
            index = 0
            while (index < len(y_result)):
                y_result[index] = y_result[index] + y_preds[clf_index][index]
                index += 1
            clf_index += 1
        # uśrednianie i zaokrąglanie
        for index, y in enumerate(y_result):
            y_result[index] = round(y_result[index] / len(clfs))
        return y_result

    def getClassificationScore(self, clf_name, y_test, y_pred):
        print("Nazwa klasyfikatora: " + clf_name)
        print(accuracy_score(y_test, y_pred))
        print(confusion_matrix(y_test, y_pred))

c=Classifiers()
X_clean = c.datasetPreprocessing(
     X = dane,columns_to_map = ['workclass','education','age','marital-status','occupation',
                                'relationship', 'race', 'sex', 'native-country', 'y'])
print(X_clean.dtypes)

X_train, X_test, y_train, y_test = c.splitDatasetIntoTrainAndTest(
      X=X_clean.drop(columns=['y']),
      y=X_clean['y'])
print(X_train)
print(y_train)


# klasyfikacja zespołowa
y_pred_ensable_train = c.ensableClassifier(
    [RandomForestClassifier(),DecisionTreeClassifier(), KNeighborsClassifier()], X_train, X_train, y_train)
y_pred_ensable_test = c.ensableClassifier(
    [RandomForestClassifier(),DecisionTreeClassifier(), KNeighborsClassifier()], X_train, X_test, y_train)
c.getClassificationScore("Uczenie zespołowe trenowanie", y_train, y_pred_ensable_train)
c.getClassificationScore("Uczenie zespołowe testowanie", y_test, y_pred_ensable_test)
#c.plotClassificationResult(X_test['age'],'age', X_test['fare'], 'fare', y_pred_ensable_test)