#!/usr/bin/env python3

#宣告 f 打開 hw2.csv
f = open('hw2.csv', 'r')

#宣告 content 讀取 f 內容
content = f.readlines()

#關閉 f
f.close()

#宣告 data_matrix 為空 list 以存取全部資料
data_matrix = []

#for 迴圈逐行處理 content:
for line in content:

    #宣告 sample 為空 list 以存取單一資料
    sample = []

    #去除換行字元 '\n'
    line = line.rstrip('\n')
    #依區隔字元 '\t' 分割字串
    line = line.split('\t')

    #宣告 one_hot 為 list 且內含 23 個 0
    one_hot = [0] * 23
    #根據分割後的字串的第一個字元 (label欄位的值) 在 one_hot 中適當位置填入 1
    one_hot[int(line[0])] = 1

    #將 one_hot 中的值放入 sample
    sample.append(one_hot)

    #for 迴圈逐字處理剩餘分割後的字串:
    for num in line[1:]:

        #如果字串 == 'none':
        if num == 'none':
            #將 0 存入 sample
            sample.append(0)
        #否則:
        else:
            #將字串轉換成數字
            num = float(num)
            #將數字存入 sample
            sample.append(num)

    #將 sample 存入 data_matrix
    data_matrix.append(sample)

print(data_matrix) 