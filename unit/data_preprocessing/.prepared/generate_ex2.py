#!/usr/bin/env python3

import numpy as np
from random import randint

def apply_zero(array):

    mask = np.random.randint(0, 49, size = array.shape) == 0
    array[mask] = np.zeros(array.shape)[mask]

if '__main__' == __name__:

    num_sample = 10000
    file_name = 'ex2.txt'
    delimiter = '\t'
    use_missing_value = True
    num_label = 23

    f1 = np.random.normal(170, 5, num_sample)
    f2 = np.random.normal(20, 3, num_sample)
    f3 = np.random.normal(50, 13, num_sample)
    f4 = np.random.normal(80, 7, num_sample)
    f5 = np.random.normal(60, 11, num_sample)

    data = np.transpose(np.stack([f1, f2, f3, f4, f5]))

    if use_missing_value:
        apply_zero(data)

    label = np.reshape(np.array([randint(0, num_label - 1) for _ in range(num_sample)]), [num_sample, 1])

    data = np.concatenate([label, data], axis = 1)

    with open(file_name, 'w') as f:

        for sample in data:
            sample = [str(int(sample[0]))] + ['%.1f' % num if num != 0 else 'none' for num in sample[1:].tolist()]

            f.write(delimiter.join(sample) + '\n')
