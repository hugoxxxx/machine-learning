# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 20:27:27 2016

@author: hugox
"""
from numpy import *
import pandas as pd
#读取西瓜数据集 3.0
df = pd.read_excel('watermelon_3.xlsx')
#获取输入属性描述个数 d
n,d = shape(df)
d = d-1
#设置隐层神经元个数 q
q = 8
#设置输出层神经元个数 l
l = 1
#创建训练数据
dataMat = array(df[['色泽', '根蒂', '敲声', '纹理', '脐部', '触感', '密度', '含糖率']].values[:,:])
labelMat = array(df['好瓜'].values[:]).reshape(n,1)
#定义学习率
alpha = 0.01
#随机初始化参数
v = random.random((d, q))
w = random.random((q, l))
theta = random.random((1,l))
gamma = random.random((1,q))

#设定计数器
count = 1
#sigmoid 函数
def sigmoid(x):
    return 1/(1+exp(-x))

#标准 BP 算法
while 1:
    bm_o = sigmoid(dot(dataMat, v) - gamma)
    ym_o = sigmoid(dot(bm_o, w) - theta)
    E = labelMat - ym_o
    if count % 10000 == 0: #每万次训练误差
        print('count=%s error=%s' %(count, sum(E**2)/n))
    if sum(E**2)/n<0.05: 
        break
    for i in range(17): #遍历训练集
        x = dataMat[i].reshape(1,d)
        y = labelMat[i].reshape(1,l)
        b_o = sigmoid(dot(x,v) - gamma)
        y_o = sigmoid(dot(b_o,w) - theta) 
        g = y_o * (1 - y_o) * (y - y_o)
        e_ = multiply((b_o * (1 - b_o)).T, dot(w, g.T)) 
        w += alpha * dot(b_o.T, g)
        theta -= alpha * g               
        v += alpha * dot(e_, x).T
        gamma -= alpha * e_.T
    count += 1
    
print('count=%s\nlabelMat.T=%s\nym_o.T=%s\n' %(count,labelMat.T,ym_o.T))