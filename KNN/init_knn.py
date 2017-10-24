import matplotlib.pyplot as plt

group1 = [(2, 1.7), (1.8, 1.8), (1.5, 1.5), (1.4, 1.8), (2, 1), (1.8, 1.2), (1.4, 2.2), (1.5, 1.3)]
group2 = [(2, 2.3), (2.3, 2.2), (2.2, 2.4), (3, 2.5), (2.8, 2.7), (2.8, 2.1), (2.6, 2.5), (2.2, 2.8)]


def train_data():
    group1 = [(1, 2, 1.7), (1, 1.8, 1.8), (1, 1.5, 1.5), (1, 1.4, 1.8), (1, 2, 1), (1, 1.8, 1.2), (1, 1.4, 2.2), (1, 1.5, 1.3)]
    group2 = [(-1, 2, 2.3), (-1, 2.3, 2.2), (-1, 2.2, 2.4), (-1, 3, 2.5), (-1, 2.8, 2.7), (-1, 2.8, 2.1), (-1, 2.6, 2.5), (-1, 2.2, 2.8)]
    train_point = group1 + group2
    return train_point


def plot(test_point):
    plt.scatter(*zip(*group1), marker='o', color='k')
    plt.scatter(*zip(*group2), marker='x', color='r')
    plt.scatter(test_point[0], test_point[1], marker=(3, 0), color='b')

    plt.ylabel("y_label")
    plt.xlabel("x_label")
    plt.title("KNN")
    plt.xlim(0.8, 3.2)
    plt.ylim(0.8, 3.2)
    plt.show()
