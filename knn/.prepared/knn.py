import math
import matplotlib.pyplot as plt

group1 = [
        (1, 2.0, 1.7),
        (1, 1.8, 1.8),
        (1, 1.5, 1.5),
        (1, 1.4, 1.8),
        (1, 2.0, 1.0),
        (1, 1.8, 1.2),
        (1, 1.4, 2.2),
        (1, 1.5, 1.3),
        ]

group2 = [
        (-1, 2.0, 2.3),
        (-1, 2.3, 2.2),
        (-1, 2.2, 2.4),
        (-1, 3.0, 2.5),
        (-1, 2.8, 2.7),
        (-1, 2.8, 2.1),
        (-1, 2.6, 2.5),
        (-1, 2.2, 2.8),
        ]

training_data = group1 + group2

def knn(tr, te, k):
    # each element in the tr is a tuple: (label, x value, y value)
    # te is a tuple: (x value, y-value)

    plot(te)

    # calculate distances of all training points to the test point
    dist = []
    for element in tr:
        # element[0] is label
        # element[1] is x value
        # element[2] is y value
        dist.append([(element[1]-te[0])**2 + (element[2]-te[1])**2, element[0]])

    # sort distances
    dist = sorted(dist)

    # calculate the label sum of the k nearest neighbors
    decision = 0
    for i in range(0, k):
        # dist[i] is the i-th nearest neighbor
        # dist[i][0] is its distance to te
        # dist[i][1] is its label
        decision = decision + dist[i][1] 
        
    if decision >= 0:
        print('The test position is in group o')
    else:
        print('The test position is in group x')

def plot(te):
    plt.scatter(*zip(*group1), marker='o', color='k')
    plt.scatter(*zip(*group2), marker='x', color='r')
    plt.scatter(te[0], te[1], marker=(3, 0), color='b')

    plt.title('knn')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(0.8, 3.2)
    plt.ylim(0.8, 3.2)
    plt.show()
