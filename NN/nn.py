import matplotlib.pyplot as plt
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

from keras.models import Sequential
from keras.layers import Dense, Activation

dataset = input_data.read_data_sets("./mnist/", one_hot = True)

class SimpleNN():

    def __init__(self):
        self.model = Sequential()
        self.layers = []

    def add_layer(self, units, activation = None):
        self.layers.append(Dense(units, input_dim = 784, activation = activation))

    def train(self, steps, optimizer = 'Adam'):
        if len(self.layers) == 0:
            self.model.add(Dense(10, input_dim = 784))
        else:
            for l in self.layers:
                self.model.add(l)

        self.model.compile(loss = 'categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
        for step in range(1, steps + 1):
            data = dataset.train.next_batch(128)
            loss, acc = self.model.train_on_batch(data[0], data[1])

            if step % 100 == 0:
                print('step: ' + str(step) + ', loss: ' + str(loss) + ', accuracy: ' + str(acc * 100) + '%')

    def predict(self):
        data = dataset.test.next_batch(100)
        pred = self.model.predict(data[0])
        mat = []
        for i in range(10):
            line = []
            for j in range(10):
                line.append(0)
            mat.append(line)

        pred = pred.tolist()
        act = data[1].tolist()

        for i in range(100):
            act_class = act[i].index(1)
            pred_class = pred[i].index(max(pred[i]))
            mat[act_class][pred_class] += 1

        plt.matshow(mat)
        plt.show()

def demo():
    nn = SimpleNN()
    nn.add_layer(256, 'relu')
    nn.add_layer(10, 'softmax')
    nn.train(2000)
    nn.predict()

if __name__ == '__main__':
    demo()

