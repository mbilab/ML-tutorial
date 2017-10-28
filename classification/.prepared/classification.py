import numpy
from IPython.display import Image 
from pydotplus import graph_from_dot_data
import os
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz

def classifier(X, y):
    clf = RandomForestClassifier(n_estimators=500, random_state=0)
    clf = clf.fit(X, y)
    return clf

def evaluate(clf, X_train, X_test, y_train, y_test):
    print('Accuracy on training samples: %s' % (clf.score(X_train, y_train)))
    print('Accuracy on test samples: %s\n' % (clf.score(X_test, y_test)))

def get_data(path):
    pwd = os.path.dirname(__file__)
    f = open(os.path.join(pwd, path), 'r')
    feature_names = f.readline().split(',')[1:]
    data = numpy.loadtxt(f, delimiter=',', dtype=int)

    random.seed(0)
    random.shuffle(data)

    return data[:, 1:], data[:, 0], feature_names

def plot(clf):
    dot_data = export_graphviz(clf, class_names=['not worthy', 'worthy'], feature_names=feature_names, filled=True, out_file=None) 
    graph = graph_from_dot_data(dot_data) 
    display(Image(graph.create_png()))

X, y, feature_names = get_data('german_credit.csv')

if '__main__' == __name__:
    pass
