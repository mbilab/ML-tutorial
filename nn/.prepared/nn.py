import matplotlib.pyplot as plt
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

from keras.models import Sequential
from keras.layers import Dense, Activation

dataset = input_data.read_data_sets("./mnist/", one_hot = True)

def get_train_data():
    data = dataset.train.next_batch(50000)
    return data[0], data[1]

def get_test_data(count = 1000):
    data = dataset.test.next_batch(count)
    return data[0], data[1]

def plot_correlation_matrix(y_true, y_pred):
    mat = []
    for i in range(10):
        line = []
        for j in range(10):
            line.append(10)
        mat.append(line)

    if type(y_true) != list:
        y_true = y_true.tolist()
    if type(y_pred) != list:
        y_pred = y_pred.tolist()

    for i in range(len(y_true)):
        t_class = y_true[i].index(1)
        p_class = y_pred[i].index(max(y_pred[i]))
        mat[t_class][p_class] += 1

    print('correlation matrix between predction and real condition: ')
    plt.matshow(mat)
    plt.show()

def demo():
    model = Sequential()
    model.add(Dense(256, input_dim = 784, activation = 'relu'))
    model.add(Dense(10, activation = 'softmax'))

    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    x_tr, y_tr = get_train_data()
    model.fit(x_tr, y_tr, epochs = 20, batch_size = 128, validation_split = 0.2, verbose = 1)

    x_te, y_te = get_test_data()
    y_pred = model.predict(x_te)

    plot_correlation_matrix(y_te, y_pred)

if __name__ == '__main__':
    demo()

