# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 11:58:38 2016

@author: hugo.xu
"""

import numpy as np
import pandas as pd

df = pd.read_excel('watermelon_3a.xlsx')
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


# 定义基学习器(决策树桩)
# 连续属性划分点（基于最小错误率）
def atrSplit(ar, status, D):
    n = status.size
    ar_sorted = np.sort(ar, axis=0)
    t, et = 0, 1
    for i in range(n-1):
        t1 = 0.5 * (ar_sorted[i] + ar_sorted[i+1])
        result_t = np.sign(ar - t1)
        er = error(result_t, status, D)
        if et > er:
            et = er
            t = t1
            result = result_t
    return t, et, result



# AdaBoost算法
def adaBoost(ar, status, T):
    table = status
    m = status.size
    D = np.repeat(1/m, m).reshape(m, 1)
    at_out = np.arange(T, dtype=float)
    t_out = np.arange(T, dtype=float)
    for i in range(T):
        t, et, result = atrSplit(ar, status, D)
        at = 0.5 * np.log((1 - et) / et)
        at_out[i] = at
        t_out[i] = t
        z = D * np.exp(-at * status * result)
        D = D / np.sum(z) * np.exp(-at * status * result)
        table = np.hstack((table, result))
        print(t)
    print(table)
    return at_out, t_out, table

adaBoost(sugar_ratio, WM_status, 3)
# 集合分类器
def classicfier(at, t, ar):
    m = at.size
    n = ar.size
    result = np.zeros(n).reshape(n, 1)
    for i in range(m):
        result_tmp = np.sign(ar - t[i])
        result += at[i] * result_tmp
    result = np.sign(result)
    return result

for T in range(10):
    at,t, table = adaBoost(sugar_ratio, WM_status, T+1)
    result = classicfier(at, t, sugar_ratio)
#    table = np.hstack((table, result))
