# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 16:45:00 2018

@author: ashwin
"""

import xgboost as xgb
import numpy as np
import pandas as pd
import time

from sklearn.model_selection import cross_val_score, train_test_split
from mlxtend.preprocessing import DenseTransformer
from sklearn.pipeline import Pipeline



# import data
train = pd.read_csv('../data/Boston/boston-train.csv', index_col = 'ID', delimiter = ',')
test = pd.read_csv('../data/Boston/boston-test.csv', index_col = 'ID', delimiter = ',')

# =============================================================================
# target = df_train['medv']
# df_train = df_train.drop(['ID','medv'],axis=1)
# 
# df_test = df_test.drop(['ID'],axis=1)
# 
# 
# xgtrain = xgb.DMatrix(df_train.values, target.values)
# xgtest = xgb.DMatrix(df_test.values)
# =============================================================================

y_price = np.log1p(train.medv)

df_train, df_test, y_train, y_test = train_test_split(train, y_price, test_size = 0.2, shuffle = True)

def defaultWithCatVars(df_train, df_test, y_train, y_test):

    dtrain = pd.get_dummies(df_train, columns= ['chas'])

    xgboost_clf = Pipeline([('to_dense', DenseTransformer()), ('clf', xgb.XGBRegressor(eval_metric = 'rmse'))])

    time_sum = 0.0
    for i in range(0,5):
        start = time.time()
        cv = cross_val_score(xgboost_clf, df_train, y_train, scoring='neg_mean_squared_error', cv=5)
        end = time.time()
        time_sum += end-start
    rmse = np.sqrt(-cv.mean())

    print ('RMSE (XGBRegressor) = {0}'.format(rmse))
    #print ("time: "+ str(end - start))
    print ("time: "+ str(time_sum/5))


# =============================================================================
# def defaultWithoutCatVars(df_train, df_test, y_train, y_test):
#     
#     df_train = df_train.drop(['chas'], axis = 1)
#     
#     xgboost_clf = Pipeline([('to_dense', DenseTransformer()), ('clf', xgb.XGBRegressor(eval_metric = 'rmse'))])
# 
#     time_sum = 0
#     for i in range(0,5):
#         start = time.time()
#         cv = cross_val_score(xgboost_clf, df_train, y_train, scoring='neg_mean_squared_error', cv=5)
#         end = time.time()
#         time_sum += end-start
#     rmse = np.sqrt(-cv.mean())
# 
#     print ('RMSE (XGBRegressor) = {0}'.format(rmse))
#     #print ("time: "+ str(end - start))
#     print ("time: "+ str(time_sum/5))
# =============================================================================
    

defaultWithCatVars(df_train, df_test, y_train, y_test)
#defaultWithoutCatVars(df_train, df_test, y_train, y_test)

