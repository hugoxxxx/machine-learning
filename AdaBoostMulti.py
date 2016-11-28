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
# 1. 连续属性划分点（基于最小错误率）
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


# 2. 定义决策树桩
def dec_stump(ar1, ar2, status, D):
    t1, et1, result1 = atrSplit(ar1, status, D)
    t2, et2, result2 = atrSplit(ar2, status, D)
    if et1 <= et2:
        t = t1
        et = et1
        result = result1
        atr = "density"
    else:
        t = t2
        et = et2
        result = result2
        atr = "sugar_ratio"
    return t, et, result, atr


# AdaBoost算法
def adaBoost(ar1, ar2, status, T):
    m = status.size
    D = np.repeat(1/m, m).reshape(m, 1)
    at_out = np.arange(T, dtype=float)
    t_out = np.arange(T, dtype=float)
    atr_out = []
    for i in range(T):
        t, et, result, atr = dec_stump(ar1, ar2, status, D)
        at = 0.5 * np.log((1 - et) / et)
        at_out[i] = at
        t_out[i] = t
        atr_out.append(atr)
        z = D * np.exp(-at * status * result)
        D = D / np.sum(z) * np.exp(-at * status * result)
        print(t)
    return at_out, t_out, atr_out


# 集合分类器
def classicfier(at, t, atr, ar1, ar2):
    m = at.size
    n = ar1.size
    result = np.zeros(n).reshape(n, 1)
    for i in range(m):
        if atr[i] == "density":
            result_tmp = np.sign(ar1 - t[i])
        else:
            result_tmp = np.sign(ar2 - t[i])
        result += at[i] * result_tmp
    result = np.sign(result)
    return result

table = WM_status
for T in range(8):
    at, t, atr = adaBoost(density, sugar_ratio, WM_status, T+1)
    result = classicfier(at, t, atr, density, sugar_ratio)
    table = np.hstack((table, result))
