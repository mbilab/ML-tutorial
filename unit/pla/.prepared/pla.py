import matplotlib.pyplot as plt
import numpy as np

def demo():
    step = 0
    # y would be the first column after rolling
    x = np.roll(np.loadtxt('./pla.dat', np.float32), 1, axis=1)
    y = np.copy(x[:,0])
    x[:,0] = np.ones(len(x)) # set the first column as bias
    w = np.zeros(x.shape[1])
    while True:
        err = 0
        for _x, _y in zip(x, y):
            if np.dot(w, _x) * _y <= 0:
                err += 1
                step += 1
                w += _x * _y
        if 0 == err: break
    print('#%d' % (step), w)

def plot(): # {{{
    size = 4
    plt.xlim(-size, size)
    plt.ylim(-size, size)
    line_x = np.array([-size, size], np.float32)
    line_y = -(w[0] + w[1] * line_x) / w[2]
    slope = -w[1] / w[2]

    # top-left or top-right
    check_point = [1, -size, size] if slope > 0 else [1, size, size]

    colors = ['r', 'b'] if np.dot(w, check_point) > 0 else ['b', 'r']
    plt.fill_between(line_x, line_y, size, alpha=0.25, color=colors[0])
    plt.fill_between(line_x, -size, line_y, alpha=0.25, color=colors[1])
    plt.plot(line_x, line_y, color='w')
    for _x, _y in zip(x, y):
        if _y > 0:
            plt.scatter(_x[0], _x[1], edgecolors='r', facecolors='none', marker='o')
        else:
            plt.scatter(_x[0], _x[1], facecolors='b', marker='x')  

    plt.title('$ \#%d: %.1f + %.1f x_1 + %.1f x_2 = 0$' % tuple(np.append(step, w)))
    plt.show()
# }}}

def reset():
    global step, w
    step = 0
    w = np.zeros(3)

def update():
    global step, w
    for _x, _y in zip(x, y):
        _x = np.append(1, _x)
        if np.dot(w, _x) * _y <= 0:
            step += 1
            w += _x * _y
            break

step = None
w = None
x = np.array([[1, 2], [2, 0], [0, 3], [2, 3], [1, 0], [-2, -2], [0, -1], [-1, 1], [-3, 1]], np.int32)
y = np.array([1, 1, 1, 1, -1, -1, -1, -1, -1], np.int32)
reset()

if '__main__' == __name__:
    pass
