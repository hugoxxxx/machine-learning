# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 20:27:27 2016

@author: hugox
"""
import numpy as np
import pandas as pd
#读取西瓜数据集 3.0
df = pd.read_excel('watermelon_3.xlsx')
#获取输入属性描述个数 d
n,d = np.shape(df)
d = d-1
#设置隐层神经元个数 q
q = 8
#设置输出层神经元个数 l
l = 1
#创建训练数据
x = np.array(df[['色泽', '根蒂', '敲声', '纹理', '脐部', '触感', '密度', '含糖率']].values[:,:])
y = np.array(df['好瓜'].values[:]).reshape(n,1)
#定义学习率
alpha = 0.01
#随机初始化参数
v = np.random.random((d, q))
w = np.random.random((q, l))
theta = np.random.random((1,l))
gamma = np.random.random((1,q))
#设定计数器
count = 1
#sigmoid 函数
def sigmoid(x):
    return 1/(1+np.exp(-x))

while 1:
    b = sigmoid(np.dot(x, v) - gamma)
    yk = sigmoid(np.dot(b, w) - theta)
    E = y - yk
    if sum(E**2)/n < 0.05:
        print('count=%s\nyk=%s' %(count, yk.T))
        break
    g = yk * (1 - yk) * E
    e =( b * (1 - b)).T * np.dot(w, g.T)
    w += alpha * np.dot(b.T, g)
    theta -= alpha * sum(g)
    v += alpha * np.dot(e, x).T
    gamma -= alpha * sum(e.T)
    count += 1