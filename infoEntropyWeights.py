# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 11:58:38 2016

@author: hugo.xu
"""

import numpy as np


# 信息熵函数
def Ent(p1, p2, y):
    if p1 == 0 and p2 != 0:
        return - p2/y * np.log2(p2/y)
    elif p1 != 0 and p2 == 0:
        return - p1/y * np.log2(p1/y)
    elif p1 == 0 and p2 == 0:
        return 0
    else:
        return - (p1/y * np.log2(p1/y) + p2/y * np.log2(p2/y))


# 信息增益函数
def gain_t(Dt1, Dt2, Dt1_T, Dt1_F, Dt2_T, Dt2_F):
    Ent_Dt1 = Ent(Dt1_T, Dt1_F, Dt1)
    Ent_Dt2 = Ent(Dt2_T, Dt2_F, Dt2)
    D = Dt1 + Dt2
    Ent_D = Ent(Dt1_T + Dt2_T, Dt1_F + Dt2_F, D)
    return Ent_D - (Dt1/D*Ent_Dt1 + Dt2/D*Ent_Dt2)


# 连续属性求划分点及信息增益
def Gain_and_T(ar_d, ar_k, n):
    Gain, T = 0, 0
    for i in range(n-1):
        ar_sorted = np.sort(ar_d, axis=0)
        t = (ar_sorted[i] + ar_sorted[i + 1])/2
        Dt1, Dt2, Dt1_T, Dt1_F, Dt2_T, Dt2_F = 0, 0, 0, 0, 0, 0
        for j in range(n):
            if ar_d[j] > t:
                if ar_k[j] == 1:
                    Dt1_T += 1
                else:
                    Dt1_F += 1
            else:
                if ar_k[j] == 1:
                    Dt2_T += 1
                else:
                    Dt2_F += 1
        Dt1 = Dt1_T + Dt1_F
        Dt2 = Dt2_T + Dt2_F
        Gain_t = gain_t(Dt1, Dt2, Dt1_T, Dt1_F, Dt2_T, Dt2_F)
        if Gain_t > Gain:
            Gain = Gain_t
            T = t
    return Gain, T
