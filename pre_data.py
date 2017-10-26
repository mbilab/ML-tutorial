import random
import numpy as np
from pprint import pprint

def readfile(filename):
    with open(filename, 'r') as f:
        rawdata = f.read()
        rawdata = rawdata.split('\n')
        del rawdata[0]
        del rawdata[-1]
    positive = []
    negative = []
    for rowdata in rawdata:
        rowdata = rowdata.split(',')
        if rowdata[0] == '1':
            positive.append(rowdata)
        else:
            negative.append(rowdata)
    return np.array(positive, dtype = np.float), np.array(negative, dtype = np.float)

def portion(data):
    temp = len([ele for ele in data if ele[0] == 1]) / len(data)
    print("Portion of Positive: {}, Portion of Negative: {}".format(temp, 1 - temp))

def get_data(num):
    if num > 900:
        num = 900
    pos, neg = readfile('data/german_credit.csv')
    #np.random.shuffle(pos)
    #np.random.shuffle(neg)
    #train500_pos = pos[:int(len(pos) * 0.5)]
    train_pos = pos[:int(num * 0.7)]
    test_pos = pos[-70:]
    #train500_neg = neg[:int(len(neg) * 0.5)]
    train_neg = neg[:int(num * 0.3)]
    test_neg = neg[-30:]
    #train500_total = np.concatenate((train500_pos, train500_neg), axis = 0)
    train_total = np.concatenate((train_pos, train_neg), axis = 0)
    test_total = np.concatenate((test_pos, test_neg), axis = 0)
    return train_total, test_total


if __name__ == "__main__":
    train, test = get_data(900)
    print("Fix train portion.")
    portion(train)
    print(len(train))
    np.save('./data/train.npy', train)
    print("Fix test portion.")
    portion(test)
    np.save('./data/test.npy', test)
    print(len(test))
