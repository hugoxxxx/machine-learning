# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 11:58:38 2016

@author: hugo.xu
"""

import numpy as np
import infoEntropy as ient
import pandas as pd

df = pd.read_excel('watermelon_3a.xlsx')

density = np.array(df[['密度']].values[:,:])
sugar_ratio = np.array(df[['含糖率']].values[:,:])
WM_status = np.array(df[['好瓜']].values[:,:])

#ar = np.sort(density, )


Gain_density, T_density = ient.Gain_and_T(density, WM_status, )

