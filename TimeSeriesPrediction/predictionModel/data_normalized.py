# -*- coding: UTF-8 -*-
# 使用LSTM进行18年并发度数据建模预测

import os
import statsmodels
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.layers import LSTM
from sklearn.externals import joblib

time_field = 'Date'
value_field = 'Value'
split_boundary = '2018-11-06 0:00:00'
# 2017年数据建模分界点
# split_boundary = '2017-11-01 1:55:00'
# filename = 'task_submit_currency_minute_all.csv'
# 2017年数据文件
filename = 'task_submit_interval2017.csv'
# filename = 'task_submit_number2017.csv'
dirs = 'midModel'

# 将数据读为一个数据框架
df = pd.read_csv(filename)
# 将时间列改为标准时间格式
# df[time_field] = pd.to_datetime(df[time_field])
# 将索引设置为时间字段
# df = df.set_index([time_field], drop=True)
max1 = df['Value'].max()
min1 = df['Value'].min()
df['Value'] = df['Value'].map(lambda x: ((x - min1) / (max1 - min1)) * 2 - 1)
print(df)
df.to_csv('task_submit_interval_normalized.csv', index=0)


# # 按2018-11-06拆分训练集和测试集
# split_date = pd.Timestamp(split_boundary)
# # 划定测试集和训练集,以分割时间为边界
# train = df.loc[:split_date]
# test = df.loc[split_date:]
# # 训练集和测试集表示在图上
# plt.figure(figsize=(10, 6))
# ax = train.plot()
# test.plot(ax=ax)
# plt.legend(['train', 'test'])
# plt.xlabel('Time')
# plt.ylabel('Currency')
# plt.savefig('currency_minute_raw.jpg', bbox_inches='tight')
# plt.show()
#
# # 将训练集和测试集缩放为[-1,1]
# # 缩放公式为：xi = (xi - mean(xi))/(max(xi)-min(xi))
# # 后续可以利用此公式进行数据还原
# # 进行数据缩放主要为了方便模型训练
# scaler = MinMaxScaler(feature_range=(-1, 1))
# train_sc = scaler.fit_transform(train)
# test_sc = scaler.transform(test)
# train_sc = train_sc[:-1]
#
#
# # 训练集和测试集表示在图上
# plt.figure(figsize=(10,6))
# plt.plot(train_sc)
# plt.show()
#
#
#
#
#
#
# # # LSTM预测
# # plt.figure(figsize=(10, 6))
# # plt.plot(y_test, label='True')
# # plt.plot(y_pred_test_lstm, label='LSTM')
# # plt.title("Currency Prediction Use LSTM")
# # plt.xlabel('Observation')
# # plt.ylabel('Currency')
# # plt.savefig('currency_minute.jpg', bbox_inches='tight')
# # plt.legend()
# # plt.show()
