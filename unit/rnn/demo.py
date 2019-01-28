#!/usr/bin/env python3
import numpy as np
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Bidirectional, SimpleRNN

def preprocessing(sequences, up=20, low=10000, maxlen=20):
    sequences = [list(filter(lambda x: x > up and x <= low, el)) for el in sequences]
    sequences = pad_sequences(sequences, maxlen=maxlen, value=low+1)
    return np.array(sequences) - up - 1

(x_train, y_train), (x_test, y_test) = imdb.load_data()

up = 20
low = 10000
maxlen = 300

x_train = preprocessing(x_train, up=up, low=low, maxlen=maxlen)
x_test = preprocessing(x_test, up=up, low=low, maxlen=maxlen)

model = Sequential()
model.add(Embedding(input_dim=low-up+1, output_dim=300, input_length=maxlen))
model.add(Bidirectional(LSTM(128)))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=200, epochs=1, validation_split=0.2)
print(model.evaluate(x_test, y_test))
