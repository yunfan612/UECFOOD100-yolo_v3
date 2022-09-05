# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 13:42:32 2022

@author: frank.chang
"""
#將label_multiple_merge檔分成train,valid,test
import random
import numpy as np

train_percent = 0.7
valid_percent = 0.2
test_percent = 0.1

path = 'D:/food_project/UECFOOD100_yoloV3/UECFOOD100/'

labelpath = path + 'label_multiple_merge.txt'
trainpath = path + 'train_list.txt'
valpath = path + 'val_list.txt'
testpath = path + 'test_list.txt'

f = open(labelpath,'r')

#line = f.readlines()
line = f.read().splitlines()
shuffle_data = random.sample(line, len(line))

train_len = int(np.ceil(len(line) * train_percent))
valid_len = int(np.ceil(len(line) * valid_percent))

train_data = shuffle_data[:train_len]
valid_data = shuffle_data[train_len : train_len + valid_len]
test_data = shuffle_data[train_len + valid_len:]


with open(trainpath, 'w') as ft:
    for item in train_data:
        ft.write("%s\n" % item)
ft.close()

with open(valpath, 'w') as fv:
    for item in valid_data:
        fv.write("%s\n" % item)
fv.close()

with open(testpath, 'w') as fte:
    for item in test_data:
        fte.write("%s\n" % item)
fte.close()
