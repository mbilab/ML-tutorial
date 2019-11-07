import numpy as np


result = np.load('.prepared/answer.npy')

def show():
    return result

def diff(request):
    count = 0
    for i in range(len(result)):
        if not np.array_equal(result[i], request[i]): count+=1
    if count == 0:
        print('Correct!')
    else:
        print('Something Wrong!')
