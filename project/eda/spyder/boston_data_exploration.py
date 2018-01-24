# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 17:05:45 2018

@author: ashwin
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv('../data/Boston/boston-train.csv', index_col = 'ID', delimiter = ',')

sns.set(color_codes=True)
#a = sns.distplot(train.medv, axlabel='Median value in $1000s.')

y_price = np.log1p(train.medv)
b = sns.distplot(y_price, axlabel='Median value in $1000s(Log Scale)', color='r')
