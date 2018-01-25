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
import util
import paramsearch
from util import plot_top_features


train = pd.read_csv('../data/Boston/boston-train.csv', index_col = 'ID', delimiter = ',')
test = pd.read_csv('../data/Boston/boston-test.csv', index_col = 'ID', delimiter = ',')

y_price = np.log1p(train.medv)

dtrain = train.drop(['medv'], axis = 1)

df_train, df_test, y_train, y_test = train_test_split(dtrain, y_price, test_size = 0.2, shuffle = True)

def defaultWithCatVars(df_train, df_test, y_train, y_test):

    dtrain = pd.get_dummies(df_train, columns= ['chas'])
    
    sk_boost_clf = Pipeline([('replace_nan', Imputer()), ('to_dense', DenseTransformer()), ('clf', GradientBoostingRegressor())])
    
    time_sum = 0.0
    for i in range(0,5):
        start = time.time()
        sklearn_cv = cross_val_score(sk_boost_clf, dtrain, y_train, scoring='neg_mean_squared_error', cv=5)
        end = time.time()
        time_sum += (end-start)
    
    print('RMSE (GradientBoostingRegressor) = {0}'.format(np.sqrt(-sklearn_cv.mean())))
    #print ("time: "+ str(end - start))
    print ("time: " + str(time_sum/5))



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
    
    

### TUNING:

# =============================================================================
# 
# def modelfit(alg, dtrain, y_price, predictors, performCV=True, printFeatureImp=True, cv_folds=5):
#     #Fitting the algorithm on the data
#     alg.fit(dtrain[predictors], y_price)
#     
#     #Predict training set:
#     dtrain_predictions = alg.predict(dtrain[predictors])
#     
#     #Cross Validation
#     if performCV:
#         cv_score = cross_val_score(sk_boost_clf, dtrain[predictors], y_price, cv=cv_folds, scoring='neg_mean_squared_error')
# 
#     #Report
#     print ("\nReport")
#     
#     
# =============================================================================



def tuned(df_train, df_test, y_train, y_test, n_folds=5):
    grid_params = {
        'clf__max_depth': [1, 2, 3],
        'clf__learning_rate': [0.1, 0.05, 0.01],
        'clf__n_estimators' : [100, 500, 1000]
    }
    
    sk_boost_clf = Pipeline([('replace_nan', Imputer()), ('to_dense', DenseTransformer()), ('clf', GradientBoostingRegressor())])
    sk_boost_clf.fit(df_train, y_train)
    
    plot_top_features(sk_boost_clf.named_steps['clf'], df_train.columns.values, 20)
    
    grid_search_sk = GridSearchCV(sk_boost_clf, grid_params, scoring='neg_mean_squared_error', cv=n_folds, verbose=True)
    grid_search_sk.fit(df_train, y_train)
    
    best_parameters_sk = max(grid_search_sk.grid_scores_, key=lambda x: x[1])[0]
    
    print (best_parameters_sk)




#### Method Calls:

#defaultWithCatVars(df_train, df_test, y_train, y_test)
#defaultWithoutCatVars(df_train, df_test, y_train, y_test)
tuned(df_train, df_test, y_train, y_test)