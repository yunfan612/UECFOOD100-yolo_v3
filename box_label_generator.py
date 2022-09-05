# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 10:44:16 2022

@author: frank.chang
"""
#讀1~100資料夾之bb_info並產生label
import os

datapath = 'D:/food_project/UECFOOD100_yoloV3/UECFOOD100'

yolo_label_filename = os.path.join(datapath, 'label.txt').replace('\\', '/')

f = open(yolo_label_filename,'w')

for id in range(100):
    print("generating %d" %(id + 1))
    
    datapath_file = os.path.join(datapath, str(id + 1)).replace('\\', '/')
    bb_filename = os.path.join(datapath_file, 'bb_info.txt').replace('\\', '/')
    
    fb = open(bb_filename,'r')
    lines = fb.readlines()[1:]
    
    for line in lines:
        str_sepa = line.split(' ')
        f.write('%s %s%s%s%s%s\n' %(datapath_file + '/' + str_sepa[0] + '.jpg', str_sepa[1] + ',', str_sepa[2] + ',', str_sepa[3] + ',', str_sepa[4].replace('\n','') + ',', id))
    
    fb.close()
f.close()
