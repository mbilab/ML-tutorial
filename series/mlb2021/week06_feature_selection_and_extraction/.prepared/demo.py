#!usr/bin/env python3


# third-party imports
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score


def select_by_test_accuracy(clf, tr_x, tr_y, te_x, te_y, remain_num):
    n_features = tr_x.shape[1]
    idx = np.arange(n_features)
    for n_remove_fea in range(0, n_features-remain_num):
        acc = []
        for i in range(0, n_features-n_remove_fea):
            _tr_x = np.delete(tr_x, i, 1)
            _te_x = np.delete(te_x, i, 1)
            clf.fit(_tr_x, tr_y)
            acc.append(accuracy_score(te_y, clf.predict(_te_x)))
        idx = np.delete(idx, np.argmax(acc))
        tr_x = np.delete(tr_x, np.argmax(acc), 1)
        te_x = np.delete(te_x, np.argmax(acc), 1)

    print('Remain features:', idx)


def select_by_cv(clf, tr_x, tr_y, remain_num):
    n_features = tr_x.shape[1]
    idx = np.arange(n_features)
    for n_remove_fea in range(0, n_features-remain_num):
        acc = []
        for i in range(0, n_features-n_remove_fea):
            _tr_x = np.delete(tr_x, i, 1)
            clf.fit(_tr_x, tr_y)
            acc.append(np.mean(cross_val_score(clf, _tr_x, tr_y, cv=10, scoring='accuracy')))
        idx = np.delete(idx, np.argmax(acc))
        tr_x = np.delete(tr_x, np.argmax(acc), 1)

    print('Remain features:', idx)


if __name__ == '__main__':
    pass
