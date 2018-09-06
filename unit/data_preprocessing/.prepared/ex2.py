#!/usr/bin/env python3

# 開啟 ex2.txt，存入 f
f = open('../ex2.txt', 'r')

# 讀取 f 內容，存入 lines
lines = f.readlines()

# 關閉 f
f.close()

# 宣告 data_matrix 為空 list 來存放結果矩陣
data_matrix = []

# for 迴圈逐行處理 lines:
for line in lines:

    # 宣告 row_vector 為空 list 來存放一筆資料
    row_vector = []

    # 去除換行字元 '\n'
    line = line.rstrip('\n')
    # 依 '\t' 分割字串
    fields = line.split('\t')

    # 宣告 one_hot 為有 23 個 0 的 list
    one_hot = [0] * 23
    # 根據第一個欄位在 one_hot 中適當位置填入 1
    one_hot[int(fields[0])] = 1

    # 將 one_hot 放入 row_vector
    row_vector.extend(one_hot)

    # for 迴圈逐欄位處理:
    for field in fields[1:]:

        # 如果字串為 'none':
        if 'none' == field:
            # 將 0 存入 row_vector
            row_vector.append(0)
        # 否則:
        else:
            # 將字串轉換成數字
            num = float(field)
            # 將數字存入 row_vector
            row_vector.append(num)

    # 將 row_vector 存入 data_matrix
    data_matrix.append(row_vector)

# 印出結果
print(data_matrix)
