import matplotlib.pyplot as plt
import numpy as np

def plot():
    size = 4
    plt.xlim(-size, size)
    plt.ylim(-size, size)
    line_x = np.array([-size, size], np.float32)
    line_y = -(w[0] + w[1] * line_x) / w[2]
    slope = -w[1] / w[2]
    if slope > 0 and np.dot(w, [0, -size, size]) > 0 or \
        slope < 0 and np.dot(w, [0, size, size]) > 0: # top > 0
        top_color, bottom_color = 'r', 'b'
    else: # top < 0
        top_color, bottom_color = 'b', 'r'
    plt.fill_between(line_x, line_y, size, alpha=0.25, color=top_color)
    plt.fill_between(line_x, -size, line_y, alpha=0.25, color=bottom_color)
    plt.plot(line_x, line_y, color='w')
    for _x, _y in zip(x, y):
        if _y > 0:
            plt.scatter(_x[0], _x[1], edgecolors='r', facecolors='none', marker='o')
        else:
            plt.scatter(_x[0], _x[1], facecolors='b', marker='x')  

    plt.title('$ \#%d: %.1f + %.1f x_1 + %.1f x_2 = 0$' % tuple(np.append(count, w)))
    plt.show()

def reset():
    global count, w
    count = 0
    w = np.array([0, 0, 0], np.float32)

def update():
    global count, w
    for _x, _y in zip(x, y):
        _x = np.append(1, _x) * 1 
        if np.dot(w, _x) * _y <= 0:
            count += 1
            w += _y * _x
            break
    plot()

count = None
w = None
x = np.array([[1, 2], [2, 0], [0, 3], [2, 3], [1, 0], [-2, -2], [0, -1], [-1, 1], [-3, 1]], np.int32)   
y = np.array([1, 1, 1, 1, -1, -1, -1, -1, -1], np.int32)
reset()

if '__main__' == __name__:
    pass
