from math import inf

import numpy as np
import tensorflow as tf

from keras.callbacks import Callback
import keras.backend as K

plt = None

def run_from_ipython():
    try:
        __IPYTHON__
        return True

    except NameError:
        return False

if run_from_ipython():
    from matplotlib import pyplot as plt

class GradientLogger(Callback):

    def __init__(self, x_eval, y_eval, layer_index=None, keep_last_epochs=3):
        self.x_eval = x_eval
        self.y_eval = y_eval
        self.sample_weight = np.ones(x_eval.shape[0])
        self.layer_index = layer_index
        self.var_name = []
        self.gradients = []
        self.keep_last_epochs = keep_last_epochs
        self.ipython_mode = run_from_ipython()

        super().__init__()

    def on_train_begin(self, logs=None):
        gradient_tensors = self.model.optimizer.get_gradients(self.model.total_loss, self.model.trainable_weights)

        input_tensors = [
            self.model.inputs[0],
            self.model.sample_weights[0],
            self.model.targets[0],
            K.learning_phase(),
        ]

        self.get_gradients = K.function(inputs=input_tensors, outputs=gradient_tensors)
        self.layer_index = self.layer_index if self.layer_index is not None else [i for i in range(len(self.model.layers))]

        for i, var in enumerate(self.model.trainable_weights):
            if 'kernel' not in var.name:
                continue

            if (i // 2) in self.layer_index:
                self.var_name.append(var.name)

    def on_train_end(self, logs=None):
        if run_from_ipython():

            f, ax = plt.subplots(len(self.layer_index), sharex=True, figsize=(8, 2.5 * len(self.layer_index)))

            for i, var_name in enumerate(self.var_name):
                grad = np.concatenate([el[i] for el in self.gradients])
                ax[i].hist(grad.flatten())
                ax[i].set_title(var_name)
            plt.show()

    def on_epoch_end(self, epoch, logs=None):
        gradients = self.get_gradients([self.x_eval, self.sample_weight, self.y_eval, 0])

        print('Gradient Distribution:')
        print('{0:<19s} {1:<12s} {2:s}'.format('Name', 'Mean', 'Std'))

        tmp = []
        for var, grad in zip(self.model.trainable_weights, gradients):
            if var.name not in self.var_name:
                continue

            print('{0:<18s} {1:>13.10f} {2:.10f}'.format(var.name, np.mean(grad), np.std(grad)))
            tmp.append(grad)

        if self.ipython_mode:
            self.gradients.append(tmp)
            self.gradients = self.gradients[-self.keep_last_epochs:]

if __name__ == '__main__':

    import numpy as np
    from keras.datasets import mnist
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.utils import to_categorical
    from keras.optimizers import SGD

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = np.reshape(x_train, [-1, 784]) / 255.0
    x_test = np.reshape(x_test, [-1, 784]) / 255.0

    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)

    model = Sequential()
    model.add(Dense(100, activation='sigmoid', input_shape=(784,)))
    model.add(Dense(100, activation='sigmoid'))
    model.add(Dense(10, activation='softmax'))

    model.summary()

    gradient_logger = GradientLogger(x_eval=x_train[:200], y_eval=y_train[:200], layer_index=[0, 1])

    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=200, epochs=10, callbacks=[gradient_logger])
    print(model.evaluate(x_test, y_test))
