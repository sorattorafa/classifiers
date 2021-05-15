from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn import tree
import sys
import numpy as np

def KNNClassifier(X_train, y_train, X_test, y_test):
        from sklearn.metrics import classification_report
        neigh = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
        neigh.fit(X_train, y_train)
        neigh.score(X_test, y_test)
        return classification_report(y_test, neigh.predict(X_test))
        
def SVMWithGridSearch(X_train, y_train, X_test, y_test):
    C_range = 2. ** np.arange(-5,15,2)
    gamma_range = 2. ** np.arange(3,-15,-2)
    #k = [ 'rbf']
    # instancia o classificador, gerando probabilidades
    srv = svm.SVC(probability=True, kernel='rbf')
    ss = StandardScaler()
    pipeline = Pipeline([ ('scaler', ss), ('svm', srv) ])
#        
    param_grid = {
        'svm__C' : C_range,
        'svm__gamma' : gamma_range
    }
#        
#        # faz a busca
    grid = GridSearchCV(pipeline, param_grid, n_jobs=-1, verbose=True)
    grid.fit(X_train, y_train)
#        
#        # recupera o melhor modelo
    model = grid.best_estimator_
    return classification_report(y_test, model.predict(X_test))
#        

def MLP(X_train, y_train, X_test, y_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
        
    clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(500, 500, 500, 500), random_state=1)
    clf.fit(X_train, y_train)
    predict = clf.predict(X_test)
    return classification_report(y_test, predict)

def RandomForestClassifiers(X_train, y_train, X_test, y_test):
    #X, y = make_classification(n_samples=1000, n_features=4, n_informative=2, n_redundant=0, random_state=0, shuffle=False)
    clf = RandomForestClassifier(n_estimators=10000, max_depth=30, random_state=1)
    clf.fit(X_train, y_train)  
    #print(clf.feature_importances_)
    predict = clf.predict(X_test)
    return classification_report(y_test, predict)
        

def decisionTree(X_train, y_train, X_test, y_test):
    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    predict = clf.predict(X_test)
    return classification_report(y_test, predict)
    #tree.plot_tree(clf)
