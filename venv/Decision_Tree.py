from clean_up import Classifiers, dane

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


# y_pred_tree_train = c.trainAndTestClassifier(DecisionTreeClassifier(), X_train,X_train,y_train)
# y_pred_tree_test = c.trainAndTestClassifier(DecisionTreeClassifier(), X_train,X_test,y_train)