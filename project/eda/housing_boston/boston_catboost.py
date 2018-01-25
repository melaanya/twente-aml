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

from catboost import Pool, CatBoostRegressor, cv
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.pylab import rcParams
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split
import util
import paramsearch
from util import plot_top_features
from util import plot_top_features, crossvaltest, catboost_param_tune


train = pd.read_csv('../data/Boston/boston-train.csv', index_col = 'ID', delimiter = ',')
df_test = pd.read_csv('../data/Boston/boston-test.csv', index_col = 'ID', delimiter = ',')

y_train = np.log1p(train.medv)

df_train = train.drop(['medv'], axis=1)

def defaultWithCatVars(df_train, y_train, n_folds=5, n_time=3):
    cat_indices = [3]
    train_pool = Pool(df_train, y_train, cat_features=cat_indices)
    # specify the training parameters 
    params = {'loss_function': 'RMSE', 'custom_metric': 'RMSE', 'eval_metric': 'RMSE', 'logging_level': 'Verbose'}
    model = CatBoostRegressor(loss_function = 'RMSE', custom_metric = 'RMSE', eval_metric= 'RMSE', calc_feature_importance = True)
    model.fit(train_pool, logging_level='Silent')
    
    plot_top_features(model, train_pool.get_feature_names(), 20)
    
    time_sum = 0
    for i in range(0,n_time):
        start = time.time()
        cv_data = cv(params, train_pool, n_folds)
        end = time.time()
        time_sum += (end-start)
    
    
    print('average cv time (CatBoost) = {0:.2f}'.format(time_sum / n_time))
    print('RMSLE (CatBoost) = {0}'.format(cv_data['RMSE_test_avg'][-1]))
    
    
#Tuned:
    
def gridSearch(df_train, y_train):
    train_pool = Pool(df_train, label = y_train)
    model = CatBoostRegressor(loss_function = 'RMSE', custom_metric = 'RMSE',  calc_feature_importance = True)
    cv_params = model.get_params()
    del cv_params['calc_feature_importance']
    
    model.fit(train_pool, logging_level='Silent')
    
    cat_grid_params = {
    'depth': [1, 2, 3],
    'learning_rate': [0.1, 0.05, 0.01],
    'iterations' : [100, 500, 1000]
    }
    
    best_params = catboost_param_tune(cat_grid_params, df_train, y_train, [3], 5)
    
    best_params['loss_function'] = 'RMSE'
    best_params['custom_metric'] = 'RMSE'
    best_params['calc_feature_importance'] = True
    print ("Best Parameters (Catboost): " + str(best_params))
    tunedWithParams(best_params, train_pool, df_train, y_train)
    
    
    
def tunedWithParams(best_params, train_pool, df_train, y_train, n_time=3, cat_indices=[3], n_folds=5):
    params = {'loss_function': 'RMSE', 'custom_metric': 'RMSE', 'eval_metric': 'RMSE', 'logging_level': 'Verbose'}

    time_sum = 0.0
    for i in range(0, n_time):
        t = time.time()
        cv_data = cv(params, train_pool, n_folds)
        time_sum += time.time() - t

    print('tuned average cv time (CatBoost) = {0:.2f}'.format(time_sum / n_time))
    print('tuned RMSLE (CatBoost) = {0}'.format(cv_data))
#Method Calls:

#defaultWithCatVars(df_train, y_train)
gridSearch(df_train, y_train)

