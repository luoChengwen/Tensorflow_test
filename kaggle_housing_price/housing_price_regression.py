import numpy as np
import tensorflow as tf
# from sklearn import
from sklearn.model_selection import train_test_split
import pandas as pd
import os
from sklearn.preprocessing import Normalizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
tf.enable_eager_execution
import matplotlib.pyplot as plt
import seaborn as sb

'''
compare regression models between nn and other ml methods
'''
train = pd.read_csv(os.getcwd() + "/kaggle_housing_price/train.csv")
test = pd.read_csv(os.getcwd() + "/kaggle_housing_price/test.csv")

train_y = train[['SalePrice']]
train_X = train.drop(columns=['SalePrice','Id'])


C_mat = train_X.corr()
fig = plt.figure(figsize = (15,15))

for i in train_X.columns:
    if train_X[i].dtype == 'O':
        dummy = pd.get_dummies(train_X[i], prefix='d_')
        pd.concat([dummy, train_X], axis=1)
        train_X.drop(columns=i)
# train_X.dropna(axis=1, inplace=True)
#
# trainX_n = train_X[train_X.dtypes[train_X.dtypes != 'O'].index]
# trainX_year = trainX_n[['YrSold']]
# trainX_n = trainX_n.drop(columns=['YrSold'], axis=1)
# trainX_n = Normalizer().fit_transform(trainX_n)
# colname = train_X[train_X.dtypes[train_X.dtypes != 'O'].index].columns
# train_X = train_X.drop(columns=colname)
# train_X = pd.concat(train_X,pd.DataFrame(trainX_n, columns=colname))
# model = Sequential()
# model.add(Dense(32, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(32, activation='relu'))
# optimizer = tf.train.AdamOptimizer(.005)
# model.compile(optimizer=optimizer, epochs=6, batch_size=40)
# model.fit(train_X,train_y)
