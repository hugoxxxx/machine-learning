# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 10:23:40 2016

@author: hugo.xu
"""

import numpy as np
import pandas as pd
import infoEntropy as en

df = pd.read_excel('watermelon_3a.xlsx')
density = np.array(df[['密度']].values[:, :])
sugar_ratio = np.array(df[['含糖率']].values[:, :])
WM_status = np.array(df[['好瓜']].values[:, :])

T = 11  # 训练轮数
m = 11  # 采样集训练样本数


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

mark = np.arange(T, dtype=float)
attribute = []
result = np.zeros(17).reshape(17, 1)
for i in range(T):
    sample = df.sample(n=m, replace=False)
    ar1 = np.array(sample[['密度']].values[:, :])
    ar2 = np.array(sample[['含糖率']].values[:, :])
    status = np.array(sample[['好瓜']].values[:, :])
    t, atr = dec_stump(ar1, ar2, status)
    mark[i] = t
    attribute.append(atr)
    if atr == "density":
        result_tmp = np.sign(density - t)
    else:
        result_tmp = np.sign(sugar_ratio - t)
    result += result_tmp
result = np.sign(result)
