# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 13:37:47 2018

@author: ashwin
"""

import numpy as np
import pandas as pd
import time

from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from mlxtend.preprocessing import DenseTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import Imputer
from sklearn import metrics
from sklearn.datasets import load_boston

import sys
sys.path.insert(0, '../')
import util
import paramsearch
from util import plot_top_features


# train = pd.read_csv('../data/Boston/boston-train.csv', index_col = 'ID', delimiter = ',')
# test = pd.read_csv('../data/Boston/boston-test.csv', index_col = 'ID', delimiter = ',')

dataset = load_boston()
df_train = pd.DataFrame(dataset.data, columns=dataset.feature_names)
y_train = np.log1p(dataset.target)

# y_train = np.log1p(train.medv)

# df_train = train.drop(['medv'], axis = 1)

#df_train, df_test, y_train, y_test = train_test_split(dtrain, y_price, test_size = 0.2, shuffle = True)

def defaultWithCatVars(df_train, y_train, n_folds=5, n_time=3):

    sk_boost_clf = Pipeline([('replace_nan', Imputer()), ('to_dense', DenseTransformer()), ('clf', GradientBoostingRegressor())])
    sk_boost_clf.fit(df_train, y_train)
    
    plot_top_features(sk_boost_clf.named_steps['clf'], df_train.columns.values, 20)
    
    time_sum = 0.0
    for i in range(0,n_time):
        start = time.time()
        sklearn_cv = cross_val_score(sk_boost_clf, df_train, y_train, scoring='neg_mean_squared_error', cv=n_folds, verbose=False)
        end = time.time()
        time_sum += (end-start)
    
    print('average cv time (GradientBoostingRegressor) = {0:.2f} sec'.format(time_sum / n_time))
    print('RMSLE (GradientBoostingRegressor) = {0}'.format(np.sqrt(-sklearn_cv.mean())))



# =============================================================================
# def defaultWithoutCatVars(df_train, df_test, y_train, y_test):
# 
#     dtrain = df_train.drop(['chas'], axis = 1)
#     
#     sk_boost_clf = Pipeline([('replace_nan', Imputer()), ('to_dense', DenseTransformer()), ('clf', GradientBoostingRegressor())])
#     
#     time_sum = 0.0
#     for i in range(0,5):
#         start = time.time()
#         sklearn_cv = cross_val_score(sk_boost_clf, dtrain, y_train, scoring='neg_mean_squared_error', cv=5)
#         end = time.time()
#         time_sum += (end-start)
#     
#     print('RMSE (GradientBoostingRegressor) = {0}'.format(np.sqrt(-sklearn_cv.mean())))
#     #print ("time: "+ str(end - start))
#     print ("time: " + str(time_sum/5))
# =============================================================================
    

#Tuning:
def gridSearch(df_train, y_train, n_folds=5):
    grid_params = {
        'clf__max_depth': [1, 2, 3],
        'clf__learning_rate': [0.1, 0.05, 0.01],
        'clf__n_estimators' : [100, 500, 1000]
    }
    sk_boost_clf = Pipeline([('replace_nan', Imputer()), ('to_dense', DenseTransformer()), ('clf', GradientBoostingRegressor())])
    sk_boost_clf.fit(df_train, y_train)
    
    grid_search_sk = GridSearchCV(sk_boost_clf, grid_params, scoring='neg_mean_squared_error', cv=n_folds, verbose=True)
    grid_search_sk.fit(df_train, y_train)
    
    best_parameters_sk = max(grid_search_sk.grid_scores_, key=lambda x: x[1])[0]
    
    print ("Best Parameters(sklearn): " + str(best_parameters_sk))
    tunedWithParams(best_parameters_sk, df_train, y_train)
    
    
    
def tunedWithParams(best_parameters_sk, df_train, y_train, n_time=3, n_folds=5):
    sk_boost_clf = Pipeline([('replace_nan', Imputer()),('to_dense', DenseTransformer()), 
                        ('clf', GradientBoostingRegressor(learning_rate = best_parameters_sk['clf__learning_rate'], 
                                                 n_estimators = best_parameters_sk['clf__n_estimators'],
                                                 max_depth = best_parameters_sk['clf__max_depth']))])

    time_sum = 0.0
    for i in range(0, n_time):
        t = time.time()
        sklearn_cv = cross_val_score(sk_boost_clf, df_train, y_train, scoring='neg_mean_squared_error', cv=n_folds, verbose=False)
        time_sum += time.time() - t

    print('tuned average cv time (GradientBoostingRegressor) = {0:.2f} sec'.format(time_sum / n_time))
    print('tuned RMSLE (GradientBoostingRegressor) = {0}'.format(np.sqrt(-sklearn_cv.mean())))




#### Method Calls:

#defaultWithCatVars(df_train, y_train)
gridSearch(df_train, y_train)