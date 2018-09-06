#!/usr/bin/env python3

import numpy as np

data_matrix = np.genfromtxt('../ex2.txt', delimiter = '\t', filling_values = 0)

label, other = np.hsplit(data_matrix, [1])
label = np.reshape(label, [-1]).astype(int)
n_label = np.unique(label).shape[0]
one_hot = np.eye(n_label)[label]

data_matrix = np.hstack([one_hot, other])

print(data_matrix)
