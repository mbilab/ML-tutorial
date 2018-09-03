#!/usr/bin/env python3

f = open('hw1.csv', 'r')

content = f.readlines()

f.close()

data_matrix = []

for line in content:

    sample = []

    line = line.rstrip('\n')
    line = line.split(',')

    one_hot = [0, 0, 0, 0]
    one_hot[int(line[0])] = 1

    sample.append(one_hot)

    for num in line[1:]:

        num = float(num)
        sample.append(num)

    data_matrix.append(sample)

print(data_matrix)
