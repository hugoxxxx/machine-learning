# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 20:27:27 2016

@author: hugox
"""
from numpy import *
import pandas as pd
#读取西瓜数据集 3.0
df = pd.read_excel('watermelon_3.xlsx')
n,d = shape(df)
#d = d - 1    #去除结果列
#con_d = 2    #连续参数列
#sep_d = d - con_d    #离散参数列
D = array(df[['色泽', '根蒂', '敲声', '纹理', '脐部', '触感','密度', 
              '含糖率','好瓜']].values[:,:])
test = array(df[['色泽', '根蒂', '敲声', '纹理', '脐部', '触感','密度', 
                 '含糖率']].values[0,:]).reshape(1,8)

#计算好瓜与否的先验概率
goodWM = 0
badWM = 0

for i in range(n):
    if D[i, 8] == 1:
        goodWM += 1
    else:
        badWM += 1

P_goodWM = (goodWM + 1)/(n + 2)
p_badWM = (badWM + 1)/(n + 2)

#计算好瓜时类条件概率
x_goodWM_sep = zeros((1,6))

for i in range(6):
    for j in range(n):
        if D[j,d-1] == 1 and test[0,i] == D[j,i]:
            x_goodWM_sep[0,i] += 1

x_goodWM_sep = x_goodWM_sep/goodWM

x_goodWM_con = zeros((1,2))

len(goodWM)

#计算非好瓜时类条件概率
x_badWM = zeros((1,6))

for i in range(6):
    for j in range(n):
        if D[j,d-1] == 0 and test[0,i] == D[j,i]:
            x_badWM[0,i] += 1
