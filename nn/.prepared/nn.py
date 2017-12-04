from keras.layers import Dense, Activation
from keras.models import Sequential
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

def demo_nn():
    model = Sequential()
    model.add(Dense(256, activation='relu', input_dim=784))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', metrics = ['accuracy'], optimizer='adam')
    model.fit(X_train, y_train, batch_size=11000, epochs=10, validation_split=0.2, verbose=1)

    y_pred = model.predict(X_test)
    #plot_correlation_matrix(y_test, y_pred)

def mnist_data():
    mnist = input_data.read_data_sets('./mnist/', one_hot=True)
    return mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

def plot_X(X, figsize=(20, 2), cols=10):
    fig = plt.figure(figsize=figsize)
    for i, v in enumerate(X.reshape(-1, 28, 28)):
        sp = fig.add_subplot(np.ceil(len(X)/cols), cols, i + 1)
        plt.imshow(v, cmap='gray')
        sp.set_title('test')
    plt.show()

X_train, y_train, X_test, y_test = mnist_data()

if '__main__' == __name__:
    demo_nn()
