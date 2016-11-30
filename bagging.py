# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 10:23:40 2016

@author: hugo.xu
"""

import numpy as np
import pandas as pd
import infoEntropy as en
import random

df = pd.read_excel('watermelon_3a.xlsx')
density = np.array(df[['密度']].values[:, :])
sugar_ratio = np.array(df[['含糖率']].values[:, :])
WM_status = np.array(df[['好瓜']].values[:, :])

m = WM_status.size  # 采样集训练样本数


# 定义决策树桩
def dec_stump(ar1, ar2, status):
    n = ar1.size
    gain1, t1 = en.Gain_and_T(ar1, status, n)
    gain2, t2 = en.Gain_and_T(ar2, status, n)
    if gain1 >= gain2:
        t = t1
        atr = "density"
    else:
        t = t2
        atr = "sugar_ratio"
    return t, atr


# bagging算法
def bagging(array1, array2, status, T):
    mark = np.arange(T, dtype=float)
    attribute = []
    result = np.zeros(m).reshape(m, 1)
    for i in range(T):
        sample = df.sample(n=m, replace=True)
        ar1 = np.array(sample[['密度']].values[:, :])
        ar2 = np.array(sample[['含糖率']].values[:, :])
        status = np.array(sample[['好瓜']].values[:, :])
        t, atr = dec_stump(ar1, ar2, status)
        mark[i] = t
        attribute.append(atr)
        if atr == "density":
            result_tmp = np.sign(array1 - t)
        else:
            result_tmp = np.sign(array2 - t)
        result += result_tmp
    result = np.sign(result)
    for i in range(m):
        if result[i] == 0:
            result[i] = np.sign(random.randint(0, 1) - 0.5)
    return result

result = WM_status
for T in range(17):
    result_T = bagging(density, sugar_ratio, WM_status, T+1)
    result = np.hstack((result, result_T))
