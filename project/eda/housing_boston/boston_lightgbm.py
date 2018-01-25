# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 20:41:42 2018

@author: ashwin
"""


import numpy as np
import pandas as pd
import time
import lightgbm as lgb

from sklearn.model_selection import cross_val_score, train_test_split
from mlxtend.preprocessing import DenseTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import Imputer, LabelEncoder



train = pd.read_csv('../data/Boston/boston-train.csv', index_col = 'ID', delimiter = ',')
test = pd.read_csv('../data/Boston/boston-test.csv', index_col = 'ID', delimiter = ',')

y_price = np.log1p(train.medv)

df_train, df_test, y_train, y_test = train_test_split(train, y_price, test_size = 0.2, shuffle = True)

#gbm = lgb.LGBMRegressor(objective='regression', num_leaves=31, learning_rate=0.05, n_estimators=20)


def defaultWithCatVars(df_train, df_test, y_train, y_test):

    df_train['chas'] = df_train['chas'].apply(LabelEncoder().fit_transform)
    
    gbm = lgb.LGBMRegressor(objective='regression')
    
    start = time.time()
    for i in range(0,5):
        lgb_cv = cross_val_score(gbm, df_train, y_train, scoring='neg_mean_squared_error', cv=5, verbose=True)
    end = time.time()
    
    print('RMSLE (LightGBM) = {0}'.format(lgb_cv['RMSE_test_avg'][-1])) 
    #print ("time: "+ str(end - start))
    print ("time: " + str((end - start)/5))



def defaultWithoutCatVars(df_train, df_test, y_train, y_test):

    df_train = df_train.drop(['chas'], axis = 1)
    
    gbm = lgb.LGBMRegressor(objective='regression')
    
    start = time.time()
    for i in range(0,5):
        lgb_cv = cross_val_score(gbm, df_train, y_train, scoring='neg_mean_squared_error', cv=5, verbose=True)
    end = time.time()
    rmse = np.sqrt(-lgb_cv.mean())
    print('RMSLE (LightGBM) = {0}'.format(rmse)) 
    #print ("time: "+ str(end - start))
    print ("time: " + str((end - start)/5))
    
    

defaultWithCatVars(df_train, df_test, y_train, y_test)
#defaultWithoutCatVars(df_train, df_test, y_train, y_test)
