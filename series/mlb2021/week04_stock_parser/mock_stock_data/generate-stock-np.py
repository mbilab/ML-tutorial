import os
import numpy
import json
import sys

path = '/home/mlb/res/stock/twse/json/'
filename_list = os.listdir(path)
filename_list.sort()

start = filename_list.index(sys.argv[1] + '.json')
end = filename_list.index(sys.argv[2] + '.json')

content = {}
for i in range(start, end + 1):
    with open(path + filename_list[i], 'r') as f:
        tmp = json.load(f)
    f.close()
    content[filename_list[i][:-5]] = tmp

stock_feature = []
answer = []
day_list = sorted(list(content.keys()))

for i in range(5, len(day_list)-1):
    # print(day_list[i])
    stock_list = sorted(list(content[day_list[i]].keys()))
    stock_list.remove('id')
    for stock in stock_list:
        if stock not in content[day_list[i+1]].keys():
            continue
        tmp = []
        # print(stock)
        for j in range(0, 5):
            if stock not in content[day_list[i-j]].keys():
                break
            for attr in ['open', 'close', 'high', 'low', 'adj_close', 'volume']:
                if content[day_list[i-j]][stock][attr] == None:
                    break
                if content[day_list[i-j]][stock][attr] == 'NULL':
                    break
                tmp.append(float(content[day_list[i-j]][stock][attr]))
        if len(tmp) != 30:
            continue
        if content[day_list[i+1]][stock]['close'] != 'NULL' and content[day_list[i+1]][stock]['close'] != None:
            #ans = 1 if float(content[day_list[i+1]][stock]['close']) - float(content[day_list[i]][stock]['close']) > 0 else 0
            ans = float(content[day_list[i+1]][stock]['close'])
            answer.append(ans)
            stock_feature.append(tmp)
        print(day_list[i])

X = numpy.array(stock_feature)
Y = numpy.array(answer)
numpy.save(sys.argv[3] + 'X', X)
numpy.save(sys.argv[3] + 'Y', Y)
