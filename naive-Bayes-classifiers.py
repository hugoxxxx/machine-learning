badWM = 0
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
P_badWM = (badWM + 1)/(n + 2)

#取得拉普拉斯修正的Ni
Ni = amax(D[:, 0:6] , axis = 0).reshape(1,6)

#计算好瓜时类条件概率
#离散属性列
x_goodWM_sep = zeros((1,6))
for i in range(6):
    for j in range(n):
        if D[j,d-1] == 1 and test[0,i] == D[j,i]:
            x_goodWM_sep[0,i] += 1
x_goodWM_sep = (x_goodWM_sep + 1)/(goodWM + Ni)

#连续属性列
mean_goodWM = D[0:8,6:8].mean(axis = 0).reshape(1,2)
std_goodWM = D[0:8,6:8].std(axis = 0, ddof = 1).reshape(1,2)
x_goodWM_con = 1/(sqrt(2*pi)*std_goodWM)*exp(-(test[0,6:8]-mean_goodWM)**2/(2*std_goodWM**2))

x_goodWM = P_goodWM * prod(x_goodWM_sep)* prod(x_goodWM_con)

#计算非好瓜时类条件概率
x_badWM_sep = zeros((1,6))

for i in range(6):
    for j in range(n):
        if D[j,d-1] == 0 and test[0,i] == D[j,i]:
            x_badWM_sep[0,i] += 1
x_badWM_sep = (x_badWM_sep + 1)/(badWM + Ni)
#连续属性列
mean_badWM = D[8:,6:8].mean(axis = 0).reshape(1,2)
std_badWM = D[8:,6:8].std(axis = 0, ddof = 1).reshape(1,2)
x_badWM_con = 1/(sqrt(2*pi)*std_badWM)*exp(-(test[0,6:8]-mean_badWM)**2/(2*std_badWM**2))
x_badWM = P_badWM * prod(x_badWM_sep)* prod(x_badWM_con)

print('好瓜的概率 = %s\n坏瓜的概率 = %s' %(x_goodWM, x_badWM))
