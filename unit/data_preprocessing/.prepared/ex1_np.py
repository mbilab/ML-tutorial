#!/usr/bin/env python3

import numpy as np

data_matrix = np.loadtxt('../ex1.csv', delimiter = ',')

label, other = np.hsplit(data_matrix, [1])
label = np.reshape(label, [-1]).astype(int)
one_hot = np.eye(4)[label]

data_matrix = np.hstack([one_hot, other])

print(data_matrix)
