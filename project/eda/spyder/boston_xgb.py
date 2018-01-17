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

df_train, df_test, y_train, y_test = train_test_split(train, train.medv, test_size = 0.2, shuffle = True)

dtrain = pd.get_dummies(df_train, columns= ['chas'])

xgboost_clf = Pipeline([('to_dense', DenseTransformer()), ('clf', xgb.XGBRegressor(eval_metric = 'rmse'))])

start = time.time()
#for i in range(0,5):
cv = cross_val_score(xgboost_clf, dtrain, y_train, scoring='neg_mean_squared_error', cv=10, verbose=True)
end = time.time()
rmse = np.sqrt(-cv.mean())

print ('RMSE (XGBRegressor) = {0}'.format(rmse))
print ("time: "+ str(end - start))
#print ("time: "+ str((end - start)/5))