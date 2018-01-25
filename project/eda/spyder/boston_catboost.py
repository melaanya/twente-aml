# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 14:53:08 2018

@author: ashwin
"""

import catboost
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import time

from catboost import Pool, CatBoostRegressor, CatBoostClassifier, cv
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.pylab import rcParams
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split
import util
import paramsearch
from util import plot_top_features
from util import plot_top_features, catboost_param_tune


train = pd.read_csv('../data/Boston/boston-train.csv', index_col = 'ID', delimiter = ',')
df_test = pd.read_csv('../data/Boston/boston-test.csv', index_col = 'ID', delimiter = ',')

dtrain = train.drop(['medv'], axis=1)

y_price = np.log1p(train.medv)

# =============================================================================
# print (list(df_train))
# print (list(df_test))
# print (list(df_train_label))
# print (y_price)
# =============================================================================


df_train, df_val, y_train, y_val = train_test_split(dtrain, y_price, test_size=0.2)

# =============================================================================
# def defaultWithoutCatVars(df_train, df_val, y_train, y_val, df_test):
#     #drop the categorical variables
#     df_train = df_train.drop(['chas'], axis = 1)
#     
#     train_pool = Pool(df_train, y_train)
#     #val_pool = Pool(X_val, cat_features=[3])
#     #test_pool = Pool(df_test, cat_features=[3])
# 
#     # specify the training parameters 
#     boston_model = CatBoostRegressor(loss_function = 'RMSE', custom_metric = 'RMSE',  eval_metric = 'RMSE', calc_feature_importance = True)
#     params = {'loss_function': 'RMSE', 'custom_metric': 'RMSE', 'eval_metric': 'RMSE', 'logging_level': 'Silent'}
#     
#     time_sum = 0
#     for i in range(0,5):
#         start = time.time()
#         scores = cv(params, train_pool, fold_count=5)
#         end = time.time()
#         time_sum += (end-start)
#     
#     #print ('RMSE_train_avg (CatBoostRegressor) = {0}'.format(scores['RMSE_train_avg'][-1]))
#     print ('RMSE_test_avg (CatBoostRegressor) = {0}'.format(scores['RMSE_test_avg'][-1]))
#     #print ('RMSE_train_stddev (CatBoostRegressor) = {0}'.format(scores['RMSE_train_stddev'][-1]))
#     #print ('RMSE_test_stddev (CatBoostRegressor) = {0}'.format(scores['RMSE_test_stddev'][-1]))
#     #print ("time: "+ str(end - start))
#     print ("time: "+ str(time_sum/5))
# =============================================================================

    #result = boston_model.predict(test_pool)
    #print("Count of trees in model = {}".format(model.tree_count_))
    #print (result)

#boston_model = CatBoostRegressor(learning_rate = 0.1, loss_function = 'RMSE', custom_metric = 'RMSE',  eval_metric = 'RMSE', calc_feature_importance = True)
#params = {'iterations':1000, 'depth': 5, 'loss_function': 'RMSE', 'custom_metric': 'RMSE', 'eval_metric': 'RMSE', 'logging_level': 'Verbose'}

# train the model
#boston_model.fit(train_pool, eval_set= val_pool, use_best_model=True, logging_level='Verbose')

# make the prediction using the resulting model
#result = boston_model.predict(test_pool)
#print("Count of trees in model = {}".format(model.tree_count_))


#print("RMLSE on val set = %f" % np.sqrt(mean_squared_error(y_val, boston_model.get_test_eval())))


def defaultWithCatVars(df_train, df_val, y_train, y_val, df_test):
    
    train_pool = Pool(df_train, y_train, cat_features= [3])
    # specify the training parameters 
    boston_model = CatBoostRegressor(loss_function = 'RMSE', custom_metric = 'RMSE',  eval_metric = 'RMSE', calc_feature_importance = True)
    params = {'loss_function': 'RMSE', 'custom_metric': 'RMSE', 'eval_metric': 'RMSE', 'logging_level': 'Verbose'}
    
    time_sum = 0
    for i in range(0,5):
        start = time.time()
        scores = cv(params, train_pool, fold_count=5)
        end = time.time()
        time_sum += (end-start)
    
    
    #print ('RMSE_train_avg (CatBoostRegressor) = {0}'.format(scores['RMSE_train_avg'][-1]))
    print ('RMSE_test_avg (CatBoostRegressor) = {0}'.format(scores['RMSE_test_avg'][-1]))
    #print ('RMSE_train_stddev (CatBoostRegressor) = {0}'.format(scores['RMSE_train_stddev'][-1]))
    #print ('RMSE_test_stddev (CatBoostRegressor) = {0}'.format(scores['RMSE_test_stddev'][-1]))
    #print ("time: "+ str(end - start))
    print ("time: "+ str(time_sum/5))   


#Tuned:
    
def tuned(df_train, y_train):
    train_pool = Pool(df_train, label = y_train, cat_features=[3])
    model = CatBoostRegressor(loss_function = 'RMSE', custom_metric = 'RMSE',  calc_feature_importance = True)
    cv_params = model.get_params()
    del cv_params['calc_feature_importance']
    
    model.fit(train_pool, logging_level='Silent')
    plot_top_features(model, train_pool.get_feature_names(), 20)
    
    cat_grid_params = {
    'depth': [1, 2, 3],
    'learning_rate': [0.1, 0.05, 0.01],
    'iterations' : [100, 500, 1000]
    }
    best_params = catboost_param_tune(cat_grid_params, df_train, y_train, [3], 5)


#Method Calls:

defaultWithCatVars(df_train, df_val, y_train, y_val, df_test)
#defaultWithoutCatVars(df_train, df_val, y_train, y_val, df_test)


