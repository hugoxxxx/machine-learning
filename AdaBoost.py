# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 11:58:38 2016

@author: hugo.xu
"""

import numpy as np
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


# 定义分类器
def classifier(x, t):
    if x > t:
        return 1
    else:
        return -1


# 定义基学习器(决策树桩)
# 1. 连续属性划分点（基于最小错误率）
def atrSplit(ar, status, D):
    n = status.size
    ar_sorted = np.sort(ar, axis=0)
    t, et = 0, 1
    for i in range(n-1):
        result = -np.ones(n)
        t1 = 0.5 * (ar_sorted[i] + ar_sorted[i+1])
        for j in range(n):
            if ar[j] > t1:
                result[j] = 1
        er = error(result, status, D)
        if et > er:
            et = er
            t = t1
    return t, et


# 2. 定义决策树桩
def dec_stump(ar1, ar2, status, D):
    t1, et1 = atrSplit(ar1, status, D)
    t2, et2 = atrSplit(ar2, status, D)
    if et1 <= et2:
        t_out = t1
        et_out = et1
    else:
        t_out = t2
        et_out = et2
    return t_out, et_out


# AdaBoost算法
def adaBoost(ar1, ar2, status, T):
    m = WM_status.size
    D = np.repeat(1/m, m)
    at_out = np.arange(T, dtype=float)
    for i in range(T):
        t, et = dec_stump(ar1, ar2, status, D)
        at = 0.5 * np.log((1 - et) / et)
        at_out[i] = at
        print(at, at_out)
    return at
       
print(adaBoost(density, sugar_ratio, WM_status, 3))
