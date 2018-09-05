#!/usr/bin/env python3
import numpy as np

data_matrix = np.genfromtxt('hw2.txt', delimiter = '\t', filling_values = 0)
print(data_matrix)
