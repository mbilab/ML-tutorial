#!/usr/bin/env python

# standard import
import json
import os
import sys

# third-party import
import numpy as np
from datetime import *

if '__main__' == __name__:
    input_date = datetime.strptime(f'{sys.argv[1]}', '%Y-%m-%d').date()
    print(f'today is {input_date}')

    ### load file ###
    files_dir = '/home/mlb/res/stock/twse/json/'
    all_files = set([filenames for (dirpath, dirnames, filenames) in os.walk(files_dir)][0])

    ### find last available file using input date ###
    n_days_ago = 1
    while True:
        curr_date_file = f'{str(input_date - timedelta(n_days_ago))}.json'
        if curr_date_file in all_files:
            print(f'last file is {curr_date_file}')
            break
        n_days_ago += 1
    data = json.load(open(f'{files_dir}{curr_date_file}')) 
    
    ### easy algorithm ###
    '''
    * this easy algorithm assume a stock will rise if yesterday rise, will fall if yesterday fall
    * replace this easy algorithm here with machine learning
    '''

    diff_set = {code: float(stock['close']) - float(stock['open']) for code, stock in data.items() if code != 'id' and stock['close'] not in {None, 'NULL'} and stock['open'] not in {None, 'NULL'}}
    diff_set_sorted = sorted(diff_set.items(), key=lambda x:abs(x[1]), reverse=True)
    decision_set = []
    top_k_stock = 3

    for (code, diff) in diff_set_sorted[:top_k_stock]:
        curr_stock_close = float(data[code]['close'])
        curr_decision = {
            "code": code,
            "life": 1,
            "type": "buy" if diff > 0 else "short",
            "weight": 1,
            "open_price": curr_stock_close,
            "close_high_price": curr_stock_close + abs(diff)/4,
            "close_low_price": curr_stock_close - abs(diff)/4  
        }
        decision_set.append(curr_decision)
    
    ### write decision ###
    if not os.path.exists('../commit/'):
        os.makedirs('../commit/')
    json.dump(decision_set, open(f'../commit/{sys.argv[1]}_{sys.argv[1]}.json', 'w'), indent=4)
