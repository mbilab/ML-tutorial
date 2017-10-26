from sklearn.ensemble import RandomForestClassifier
import pre_data as pd

def model(num=900):
    raw_train, raw_test = pd.get_data(num)
    X_train = raw_train[:, 1:]
    Y_train = raw_train[:, 0]
    X_test = raw_test[:, 1:]
    Y_test = raw_test[:, 0]
    clf_RF = RandomForestClassifier(n_estimators = 80, max_depth = 3, min_samples_leaf = 0.01, random_state = 0)
    clf_RF = clf_RF.fit(X_train, Y_train)
    print("Model: Random Forest")
    print("Number of samples: {}".format(num))
    print("Accuracy on Training samples: {}".format(clf_RF.score(X_train, Y_train)))
    print("Accuracy on Testing samples: {}".format(clf_RF.score(X_test, Y_test)))
    #print("Average accuracy of 10 RFs: {}".format(np.sum(cross_val_score(clf_RF, X_train, Y_train, cv=10) / 10)))

