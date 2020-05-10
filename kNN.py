from clean_up import Classifiers, dane
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier



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



def trainAndTestClassifier(clf, X_train, X_test, y_train):
    print(clf)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return y_pred


def getClassificationScore(clf_name, y_test, y_pred):
    print("Nazwa klasyfikatora: " + clf_name)
    print(accuracy_score(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

y_pred_knn3_train = trainAndTestClassifier(KNeighborsClassifier(n_neighbors=3), X_train,X_train,y_train)
y_pred_knn3_test = trainAndTestClassifier(KNeighborsClassifier(n_neighbors=3), X_train,X_test,y_train)
y_pred_knn5_train = trainAndTestClassifier(KNeighborsClassifier(n_neighbors=5), X_train,X_train,y_train)
y_pred_knn5_test = trainAndTestClassifier(KNeighborsClassifier(n_neighbors=5), X_train,X_test,y_train)
y_pred_knn7_train = trainAndTestClassifier(KNeighborsClassifier(n_neighbors=7), X_train,X_train,y_train)
y_pred_knn7_test = trainAndTestClassifier(KNeighborsClassifier(n_neighbors=7), X_train,X_test,y_train)
getClassificationScore("kNN-3 trenowanie", y_train, y_pred_knn3_train)
getClassificationScore("kNN-3 testowanie", y_test, y_pred_knn3_test)
getClassificationScore("kNN-5 trenowanie", y_train, y_pred_knn5_train)
getClassificationScore("kNN-5 testowanie", y_test, y_pred_knn5_test)
getClassificationScore("kNN-7 trenowanie", y_train, y_pred_knn7_train)
getClassificationScore("kNN-7 testowanie", y_test, y_pred_knn7_test)