#!/usr/bin/env python3

import numpy as np

data_matrix = np.loadtxt('../ex1.csv', delimiter = ',')

label, other = np.split(data_matrix, [1], axis = 1)
label = np.reshape(label, [-1]).astype(int)

one_hot = np.eye(4)[label]

data_matrix = np.concatenate([one_hot, other], axis = 1)
print(data_matrix)
