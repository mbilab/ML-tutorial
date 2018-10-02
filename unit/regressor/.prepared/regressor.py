import numpy as np
from os import path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score

def load_data():
    import numpy as np
    import pandas as pd

    df = pd.read_csv('./breast_cancer_data.csv')
    mapping_dictionary = {'diagnosis':{'M': 1, 'B': 0}}
    df = df.replace(mapping_dictionary)
    df = df.drop('id', axis = 1)

    df_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
    train_X, train_Y = df_train.iloc[:, 1:].values.astype(np.float64), df_train.iloc[:, 0].values.astype(np.float64)
    test_X, test_Y = df_test.iloc[:, 1:].values.astype(np.float64), df_test.iloc[:, 0].values.astype(np.float64)

    return train_X, train_Y, test_X, test_Y

def demo():
    # liner regression thr = 0.4, acc > 0.93
    thr = 0.5
    train_X, train_Y, test_X, test_Y = load_data()
    clf = Pipeline([('scl', StandardScaler()),
                    ('pca', PCA(n_components=2)),
                    ('clf', SVR())])
    clf = clf.fit(train_X, train_Y)
    y_pred = clf.predict(test_X)

    y_pred[y_pred >= thr] = 1
    y_pred[y_pred < thr] = 0

    return y_pred

def evaluate(y_true, y_pred):
    print("Test Accuracy: {:.3f}".format(accuracy_score(y_true, y_pred)))


if '__main__' == __name__:
    demo()
