#!/usr/bin/env python3

f = open('hw2.csv', 'r')

content = f.readlines()

f.close()

data_matrix = []

for line in content:

    sample = []

    line = line.rstrip('\n')
    line = line.split('\t')

    one_hot = [0] * 23
    one_hot[int(line[0])] = 1

    sample.append(one_hot)

    for num in line[1:]:

        if num == 'none':
            sample.append(0)
        else:
            num = float(num)
            sample.append(num)

    data_matrix.append(sample)

print(data_matrix)

