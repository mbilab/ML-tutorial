#!/usr/bin/env python3

import keras
from keras.layers import Dense
from keras.models import Sequential
import numpy as np
from sklearn.metrics import accuracy_score

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

model = Sequential()
model.add(Dense(100, activation='sigmoid', input_shape=(784,)))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(mnist.train.images, mnist.train.labels, epochs=3, batch_size=100, verbose=1)

print('Test Accuracy:', accuracy_score(np.argmax(mnist.test.labels, axis=1), np.argmax(model.predict(mnist.test.images), axis=1)))
