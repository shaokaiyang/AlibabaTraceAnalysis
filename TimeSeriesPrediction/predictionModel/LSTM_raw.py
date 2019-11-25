# -*- coding: UTF-8 -*-
# 使用LSTM进行18年并发度数据建模预测

import os
import math
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
filename = 'task_submit_currency_minute_all.csv'
# 2017年数据文件
# filename = 'task_submit_interval2017.csv'
# filename = 'task_submit_number2017.csv'
dirs = 'midModel'

# 将数据读为一个数据框架
df = pd.read_csv(filename)
# 将时间列改为标准时间格式
df[time_field] = pd.to_datetime(df[time_field])
# print(df[time_field])
# 将索引设置为时间字段
df = df.set_index([time_field], drop=True)
# 原始数据折线图表示
plt.figure(figsize=(30, 6))
df[value_field].plot()
plt.show()

# 按2018-01-01拆分训练集和测试集
split_date = pd.Timestamp(split_boundary)
# 划定测试集和训练集,以分割时间为边界
train = df.loc[:split_date]
test = df.loc[split_date:]


# 将训练集和测试集缩放为[-1,1]
# 缩放公式为：xi = (xi - mean(xi))/(max(xi)-min(xi))
# 后续可以利用此公式进行数据还原
# 进行数据缩放主要为了方便模型训练
scaler = MinMaxScaler(feature_range=(-1, 1))
train_sc = scaler.fit_transform(train)
test_sc = scaler.transform(test)
print(test_sc)
print(train_sc)
# 训练集取除去最后一个元素的所有值
X_train = train_sc[:-1]
# 训练集取去除第一个元素的所有值
y_train = train_sc[1:]
# X_test = test_sc[:-1]
X_test = test_sc[1:]
# X_test = test_sc
y_test = test_sc[1:]

# 通过下面方式进行模型调用
# LSTM = joblib.load(dirs + '/LSTM.pkl')

# LSTM
# 进行LSTM输入数据的变换
X_train_lstm = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test_lstm = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

lstm_model = Sequential()
# 有一个可见层，有一个输入，隐藏层有7个LSTM神经元，输出层进行单值预测，使用relu函数进行激活
lstm_model.add(LSTM(7, input_shape=(1, X_train_lstm.shape[1]), activation='relu', kernel_initializer='lecun_uniform',
                    return_sequences=False))
lstm_model.add(Dense(1))
lstm_model.compile(loss='mean_squared_error', optimizer='adam')
early_stop = EarlyStopping(monitor='loss', patience=2, verbose=1)
history_lstm_model = lstm_model.fit(X_train_lstm, y_train, epochs=100, batch_size=50, verbose=1, shuffle=False,
                                    callbacks=[early_stop])
# 模型保存
# joblib.dump(lstm_model, dirs+'/LSTM_SECOND.model')

# lstm_model = joblib.load(dirs + '/LSTM_MINUTE.model')

# 预测值
# 用test集合进行预测
y_pred_test_lstm = lstm_model.predict(X_test_lstm)
np.savetxt('LSTM3.csv', y_pred_test_lstm, delimiter=',')

# LSTM_prediction = test.copy()
# LSTM_prediction['Value'] = y_pred_test_lstm


# LSTM预测
plt.figure(figsize=(15, 8))
plt.plot(y_test, label='True')
plt.plot(y_pred_test_lstm, label='LSTM')
plt.xlabel('Observation', fontname='Arial', fontsize=22)
plt.ylabel('Concurrency', fontname='Arial', fontsize=22)
plt.legend(loc='upper right', fontsize=22)
plt.tick_params(labelsize=18)
plt.savefig('currency_minute_LSTM.jpg', bbox_inches='tight')
plt.show()


# plt.figure(figsize=(15, 8))
# plt.plot(train['Value'], label='Train')
# plt.plot(test['Value'], label='Test')
# plt.plot(currency_simple_avg['Value'], label='Interval_Average')
# plt.xlabel('Time', fontname='Arial', fontsize=22)
# plt.ylabel('Currency', fontname='Arial', fontsize=22)
# plt.legend(loc='upper right', fontsize=22)
# plt.tick_params(labelsize=18)
# plt.savefig('simple1_average_all.jpg', bbox_inches='tight')
# plt.show()


R2 = r2_score(y_test, y_pred_test_lstm)
MSE = mean_squared_error(y_test, y_pred_test_lstm)
RMSE = math.sqrt(MSE)
MeanAE = mean_absolute_error(y_test, y_pred_test_lstm)
MedianAE = median_absolute_error(y_test, y_pred_test_lstm)
print("R2: %s" % R2)
print("MSE: %s" % MSE)
print("RMSE: %s" % RMSE)
print("MeanAE: %s" % MeanAE)
print("MedianAE: %s" % MedianAE)