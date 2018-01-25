# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 16:45:00 2018

@author: ashwin
"""

import xgboost as xgb
import numpy as np
import pandas as pd
import time

from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from mlxtend.preprocessing import DenseTransformer
from sklearn.pipeline import Pipeline
import util
import paramsearch
from util import plot_top_features



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

dtrain = train.drop(['medv'], axis = 1)

df_train, df_test, y_train, y_test = train_test_split(dtrain, y_price, test_size = 0.2, shuffle = True)

def defaultWithCatVars(df_train, df_test, y_train, y_test):

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
    


def tuned(df_train, df_test, y_train, y_test, n_folds=5):
    grid_params = {
        'clf__max_depth': [1, 2, 3],
        'clf__learning_rate': [0.1, 0.05, 0.01],
        'clf__n_estimators' : [100, 500, 1000]
    }
    
    xgboost_clf = Pipeline([('to_dense', DenseTransformer()), ('clf', xgb.XGBRegressor(eval_metric = 'rmse'))])
    xgboost_clf.fit(df_train, y_train)
    
    plot_top_features(xgboost_clf.named_steps['clf'], df_train.columns.values, 20)
    
    grid_search_xgb = GridSearchCV(xgboost_clf, grid_params, scoring='neg_mean_squared_error', cv=n_folds, verbose=True)
    grid_search_xgb.fit(df_train, y_train)
    
    best_parameters_xgb = max(grid_search_xgb.grid_scores_, key=lambda x: x[1])[0]
    
    print (best_parameters_xgb)


#### Method Calls:

#defaultWithCatVars(df_train, df_test, y_train, y_test)
#defaultWithoutCatVars(df_train, df_test, y_train, y_test)
tuned(df_train, df_test, y_train, y_test)