import numpy as np
import math
import init_knn



def demo():
    train_point = init_knn.train_data()
    test_point = (2, 2)
    init_knn.plot(test_point)
    k = 5
    distance_group = []

    # 計算所有點與測試點的距離
    # point example: (label, x value, y value)
    for point in train_point:
        distance_group.append([math.sqrt((point[1]-test_point[0])**2 + (point[2]-test_point[1])**2), point[0]])

    # 利用 sorted 這個函數，我們可以依照距離來由小到大排序
    distance_group = sorted(distance_group)

    # 計算前 K 名屬於 group1, group2 的數量
    decision = 0
    for i in range(0, k):
        decision = decision + distance_group[i][1] 
        
    # 若屬於 group1 的數量多於 group2，則判定測試點為 group1; 反之則為 group2
    if decision >= 0: print('test point is in group o')
    else: print('test point is in group x')
