#!/usr/bin/env python3
import numpy as np
from math import sqrt, pi, exp

class MyGaussianNB(object):
    def __init__(self):
        pass


    def split_dataset(self, x, y):
        self.train_data = {}
        for i in range(len(y)):
            try:
                self.train_data[y[i]].append(list(x[i]))
            except:
                self.train_data[y[i]] = [list(x[i])]


    def compute_gaussian_params(self, data):
        return [(np.mean(data[:, 0]), np.std(data[:, 0])), (np.mean(data[:, 1]), np.std(data[:, 1]))]


    def fit(self, x, y):
        self.split_dataset(x, y)
        self.gaussian_params = {}
        for label, data in self.train_data.items():
            self.gaussian_params[label] = self.compute_gaussian_params(np.array(data))


    def probability_for_each_class(self, x):
        n = sum([len(i) for i in self.train_data.values()])
        probs = [0]*len(self.train_data.keys())
        for k in self.train_data.keys():
            probs[k] = len(self.train_data[k])/n
            for i in range(len(x)):
                mean, var = self.gaussian_params[k][i]
                probs[k] *= self.calculate_probability(x[i], mean, var)

        return probs


    def predict(self, x):
        y_pred = []
        for data in x:
            class_prob = self.probability_for_each_class(data)
            y_pred.append(np.argmax(class_prob))
        return np.array(y_pred)


    def calculate_probability(self, x, mean, stdev):
        exponent = exp(-((x-mean)**2 / (2 * stdev**2 )))
        return (1 / (sqrt(2 * pi) * stdev)) * exponent
