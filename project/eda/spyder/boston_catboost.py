# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 14:53:08 2018

@author: ashwi
"""

import catboost
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from catboost import Pool, CatBoostRegressor, CatBoostClassifier, cv
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.pylab import rcParams
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split


df_train = pd.read_csv('../data/Boston/boston-train.csv', index_col = 'ID', delimiter = ',')
df_test = pd.read_csv('../data/Boston/boston-test.csv', index_col = 'ID', delimiter = ',')

df_train_label = df_train[['medv']].copy()

y_price = np.log1p(df_train.medv)

# =============================================================================
# print (list(df_train))
# print (list(df_test))
# print (list(df_train_label))
# print (y_price)
# =============================================================================


X_train, X_val, y_train, y_val = train_test_split(df_train, y_price, test_size=0.2)

print (X_train.shape, y_train.shape)
print (X_val.shape, y_val.shape)

#4


train_pool = Pool(X_train, y_train, cat_features= [3])
val_pool = Pool(X_val, cat_features=[3])
test_pool = Pool(df_test, cat_features=[3])



# specify the training parameters 
boston_model = CatBoostRegressor(learning_rate = 0.1, loss_function = 'RMSE', custom_metric = 'RMSE',  eval_metric = 'RMSE', calc_feature_importance = True)


params = {'iterations':1000, 'depth': 5, 'loss_function': 'RMSE', 'custom_metric': 'RMSE', 'eval_metric': 'RMSE', 'logging_level': 'Verbose'}
scores = cv(params, train_pool, fold_count=5)
print (len(scores['RMSE_test_avg']))
print (scores['RMSE_test_avg'][-2])



# train the model
#boston_model.fit(train_pool, eval_set= val_pool, use_best_model=True, logging_level='Verbose')

# make the prediction using the resulting model
#result = boston_model.predict(test_pool)
#print("Count of trees in model = {}".format(model.tree_count_))


#print("RMLSE on val set = %f" % np.sqrt(mean_squared_error(y_val, boston_model.get_test_eval())))



