import random
import numpy as np

def readdata(filename):
    with open(filename, 'r') as f:
        rawdata = f.read().split('\n')
    del rawdata[-1]
    return np.array([data.split(" ") for data in rawdata], dtype = np.float32)

def sign(num):
    if num > 0:
        return 1
    else:
        return -1

def accuracy(W, datas):
    pred = np.array([sign(np.matmul(W, np.append(data[:4], 1))) for data in datas])
    return np.sum(np.array(pred == datas[:, -1], dtype = np.float32)) / len(datas)

if __name__ == "__main__":
    datas = readdata("pla.dat")
    step = 0
    #for j in range(2000):
    c = 0
    lr = 1
    W = np.array([0, 0, 0, 0, 0], dtype = np.float32)
    #np.random.shuffle(datas)
    done = False
    while not done:
        err = 0
        for data in datas:
            if sign(np.matmul(W, np.append(data[:4], 1))) != data[-1]:
                W = W + lr * data[-1] * np.append(data[:4], 1)
                c += 1
                err += 1
        if err == 0:
            done = True
    print(W)
    #    step += c
    #print("Iter: {}, W: {}, acc: {}".format(c, W, accuracy(W, datas)))
    #print(step / 2000)
