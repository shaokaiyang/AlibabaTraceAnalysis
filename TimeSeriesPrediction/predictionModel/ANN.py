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
print(df[time_field])
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
# 训练集和测试集表示在图上
plt.figure(figsize=(30, 6))
ax = train.plot()
test.plot(ax=ax)
plt.legend(['train', 'test'])
plt.xlabel('Time')
plt.ylabel('Currency')
plt.savefig('currency_minute_raw.jpg', bbox_inches='tight')
plt.show()

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

# 用于时间序列预测的简单人工神经网络ANN
# 创建一个序列模型
nn_model = Sequential()
# 通过add（）方法添加层，将input_dim参数传递到第一层，激活函数为线性整流函数relu
nn_model.add(Dense(12, input_dim=1, activation='relu'))
nn_model.add(Dense(1))
# 完成学习过程配置，损失函数是mean_squared_error，优化器是adam
nn_model.compile(loss='mean_squared_error', optimizer='adam')
# 监测到loss停止改进时结束训练，patience表示经过该周期依旧没有改进可以结束训练
early_stop = EarlyStopping(monitor='loss', patience=2, verbose=1)
# 人工神经网络训练周期为100个周期，每次用一个样本进行训练
history = nn_model.fit(X_train, y_train, epochs=100, batch_size=1, verbose=1, callbacks=[early_stop], shuffle=False)
# 根据训练好的模型预测结果
# 用test集合进行预测
y_pred_test_nn = nn_model.predict(X_test)
np.savetxt('ANN.csv', y_pred_test_nn, delimiter=',')


# 保存训练好的模型
dirs = 'midModel'
if not os.path.exists(dirs):
    os.makedirs(dirs)
joblib.dump(nn_model, dirs+'/ANN.pkl')
# 通过下面方式进行模型调用
# ANN = joblib.loab(dirs + '/ANN.pkl')


# LSTM预测
plt.figure(figsize=(15, 8))
plt.plot(y_test, label='True')
plt.plot(y_pred_test_nn, label='ANN')
plt.xlabel('Observation',fontname='Arial', fontsize=22)
plt.ylabel('Concurrency',fontname='Arial', fontsize=22)
plt.legend(loc='upper right', fontsize=22)
plt.tick_params(labelsize=18)
plt.savefig('ANN.jpg', bbox_inches='tight')
plt.show()




R2 = r2_score(y_test, y_pred_test_nn)
MSE = mean_squared_error(y_test, y_pred_test_nn)
RMSE = math.sqrt(MSE)
MeanAE = mean_absolute_error(y_test, y_pred_test_nn)
MedianAE = median_absolute_error(y_test, y_pred_test_nn)
print("R2: %s" % R2)
print("MSE: %s" % MSE)
print("RMSE: %s" % RMSE)
print("MeanAE: %s" % MeanAE)
print("MedianAE: %s" % MedianAE)

