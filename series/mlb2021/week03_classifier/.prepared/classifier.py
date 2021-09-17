from IPython.display import Image
import numpy
from os import path
from pydotplus import graph_from_dot_data
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz

def demo(x, y):
    clf = RandomForestClassifier(n_estimators=500, random_state=0)
    clf = clf.fit(x, y)
    return clf

def evaluate(clf, x_tr=None, x_te=None, y_tr=None, y_te=None):
    if x_tr is None:
        x_tr = x_train
    if x_te is None:
        x_te = x_test
    if y_tr is None:
        y_tr = y_train
    if y_te is None:
        y_te = y_test
    print('Accuracy on training samples: %s' % (clf.score(x_train, y_train)))
    print('Accuracy on test samples: %s\n' % (clf.score(x_test, y_test)))

def german_credit_data():
    pwd = path.dirname(__file__)
    f = open(path.join(pwd, 'german_credit.csv'), 'r')
    feature_names = f.readline().split(',')[1:]
    data = numpy.loadtxt(f, delimiter=',', dtype=int)

    return train_test_split(
        data[:, 1:],
        data[:, 0],
        random_state=0,
        test_size=0.2
    ) + [feature_names]

def plot(clf):
    dot_data = export_graphviz(clf, class_names=['not worthy', 'worthy'], feature_names=feature_names, filled=True, out_file=None)
    graph = graph_from_dot_data(dot_data)
    display(Image(graph.create_png()))

x_train, x_test, y_train, y_test, feature_names = german_credit_data()

if '__main__' == __name__:
    pass
