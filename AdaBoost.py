# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 11:58:38 2016

@author: hugo.xu
"""

import numpy as np
import infoEntropy as ient
import pandas

df = pandas.read_excel('watermelon_3a.xlsx')
density = np.array(df[['密度']].values[:, :])
sugar_ratio = np.array(df[['含糖率']].values[:, :])
WM_status = np.array(df[['好瓜']].values[:, :])


# 错误率计算
def error(ar, status, D):
    n = status.size
    er = 0
    for i in range(n):
        if ar[i] != status[i]:
            er += D[i]
    return er


# 定义基学习器
def dec_stump(ar1, ar2, status, D):
    n = status.size
    result = -np.ones(n)
    ar1_gain, ar1_T = ient.Gain_and_T(ar1, status, n)
    ar2_gain, ar2_T = ient.Gain_and_T(ar2, status, n)
    if ar1_gain > ar2_gain:
        ar, T = ar1, ar1_T
    else:
        ar, T = ar2, ar2_T
    for i in range(n):
        if ar[i] > T:
            result[i] = 1
    et = error(result, status, D)
    print(ar1_T, ar2_T, result, et)
    return result


# AdaBoost算法
#def adaBoost():
#    D = 1/m
    
D = np.repeat(1/10, 17)            
   
dec_stump(density, sugar_ratio, WM_status, D)
