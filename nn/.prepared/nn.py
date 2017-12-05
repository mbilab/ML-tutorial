from keras.layers import Dense, Activation
from keras.models import Sequential
from keras.utils import to_categorical
from keras.utils.data_utils import get_file
import matplotlib.pyplot as plt
import numpy as np

def demo_mnist_nn():
    model = Sequential()
    model.add(Dense(256, activation='relu', input_dim=784))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', metrics = ['accuracy'], optimizer='adam')
    model.fit(x_train, y_train, batch_size=10000, epochs=10, verbose=0)
    return model

def mnist_data():
    # download from aws
    path = get_file(
            'mnist.npz',
            'https://s3.amazonaws.com/img-datasets/mnist.npz',
            cache_dir='.',
            cache_subdir='.',
            file_hash='8a61469f7ea1b51cbae51d4f78837e45'
            )

    # load and normalize
    f = np.load(path)
    x_train, y_train = f['x_train'].reshape(-1, 784) / 255, to_categorical(f['y_train'], 10)
    x_test, y_test = f['x_test'].reshape(-1, 784) / 255, to_categorical(f['y_test'], 10)
    f.close()

    return x_train, y_train, x_test, y_test

def plot_images(x, y, figsize=(15, 1.5), cols=10):
    plt.figure(figsize=figsize)
    for i, v in enumerate(x.reshape(-1, 28, 28)):
        plt.subplot(np.ceil(len(x)/cols), cols, i + 1)
        plt.axis('off')
        plt.title(y[i].argmax())
        plt.imshow(v, cmap='gray')
    plt.show()

x_train, y_train, x_test, y_test = mnist_data()

if '__main__' == __name__:
    demo_nn()
