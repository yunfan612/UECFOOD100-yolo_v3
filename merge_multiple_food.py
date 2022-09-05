# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 09:50:45 2022

@author: frank.chang
"""
#讀label,將重複檔案但不同box的label merge起來,產生label_multiple_merge
#from more_itertools import sort_together
import numpy as np
import copy

path = 'D:/food_project/UECFOOD100_yoloV3/UECFOOD100/label.txt'
merge_path = 'D:/food_project/UECFOOD100_yoloV3/UECFOOD100/label_multiple_merge.txt'

f = open(path,'r')

line = f.read().splitlines()
f.close()

sort_data = []
sort_data_true = []
#sort_data.append(1)

for i in range (len(line)):
    data_name = line[i].split('.')
    data_name = data_name[0].split('/')
    data_name = data_name[-1]
    
    sort_data.append(int(data_name))
    

key_rank = np.argsort(sort_data)   

for j in range (len(line)):
    sort_data_true.append(line[key_rank[j]])
    

idx = 0
name = 1
remove_row = []
sort_data_true_merge = copy.deepcopy(sort_data_true)

for k in range (1, len(sort_data_true)):
    name_tmp = sort_data_true[k].split('.')
    name_tmp = name_tmp[0].split('/')
    name_tmp = name_tmp[-1]
    
    if int(name_tmp) == name:
        add_box = sort_data_true[k].split(' ')
        sort_data_true_merge[idx] = sort_data_true_merge[idx] + ' ' + add_box[1]
        remove_row.append(k)
    else:
        name = int(name_tmp)
        idx = k
        
#del sort_data_true_merge[remove_row]
remove_row.reverse()
for remove_idx in remove_row:
    del sort_data_true_merge[remove_idx]
        
with open(merge_path, 'w') as fm:
    for item in sort_data_true_merge:
        fm.write("%s\n" % item)
fm.close()

#s = sort_together([key_rank, line])[1]

