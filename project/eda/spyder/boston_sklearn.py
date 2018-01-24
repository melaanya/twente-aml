# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 13:37:47 2018

@author: ashwin
"""

import numpy as np
import pandas as pd
import time

from sklearn.model_selection import cross_val_score, train_test_split
from mlxtend.preprocessing import DenseTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import Imputer



train = pd.read_csv('../data/Boston/boston-train.csv', index_col = 'ID', delimiter = ',')
test = pd.read_csv('../data/Boston/boston-test.csv', index_col = 'ID', delimiter = ',')

y_price = np.log1p(train.medv)

df_train, df_test, y_train, y_test = train_test_split(train, y_price, test_size = 0.2, shuffle = True)

def defaultWithCatVars(df_train, df_test, y_train, y_test):

    dtrain = pd.get_dummies(df_train, columns= ['chas'])
    
    sk_boost_clf = Pipeline([('replace_nan', Imputer()), ('to_dense', DenseTransformer()), ('clf', GradientBoostingRegressor())])
    
    start = time.time()
    for i in range(0,5):
        sklearn_cv = cross_val_score(sk_boost_clf, dtrain, y_train, scoring='neg_mean_squared_error', cv=3, verbose=True)
    end = time.time()
    
    print('RMSE (GradientBoostingRegressor) = {0}'.format(np.sqrt(-sklearn_cv.mean())))
    #print ("time: "+ str(end - start))
    print ("time: " + str((end - start)/5))



def defaultWithoutCatVars(df_train, df_test, y_train, y_test):

    dtrain = df_train.drop(['chas'], axis = 1)
    
    sk_boost_clf = Pipeline([('replace_nan', Imputer()), ('to_dense', DenseTransformer()), ('clf', GradientBoostingRegressor())])
    
    start = time.time()
    for i in range(0,5):
        sklearn_cv = cross_val_score(sk_boost_clf, dtrain, y_train, scoring='neg_mean_squared_error', cv=3, verbose=True)
    end = time.time()
    
    print('RMSE (GradientBoostingRegressor) = {0}'.format(np.sqrt(-sklearn_cv.mean())))
    #print ("time: "+ str(end - start))
    print ("time: " + str((end - start)/5))
    
    

defaultWithCatVars(df_train, df_test, y_train, y_test)
#defaultWithoutCatVars(df_train, df_test, y_train, y_test)
