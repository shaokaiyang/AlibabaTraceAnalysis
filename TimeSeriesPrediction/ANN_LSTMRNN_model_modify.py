# -*- coding: UTF-8 -*-
# ANN 人工神经网络；LSTM RNN 长短期记忆循环神经网络

import os
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.layers import LSTM
from sklearn.externals import joblib

time_field = 'Date'
value_field = 'Value'
split_boundary = '2018-01-01'
filename = 'stock_data_modify.csv'

# 将数据读为一个数据框架
df = pd.read_csv(filename)
# 将时间列改为标准时间格式
df[time_field] = pd.to_datetime(df[time_field])
# 将索引设置为时间字段
df = df.set_index([time_field], drop=True)
# 原始数据折线图表示
plt.figure(figsize=(10, 6))
df[value_field].plot()
plt.show()

# 按2018-01-01拆分训练集和测试集
split_date = pd.Timestamp(split_boundary)
# 划定测试集和训练集,以分割时间为边界
train = df.loc[:split_date]
test = df.loc[split_date:]
# 训练集和测试集表示在图上
plt.figure(figsize=(10, 6))
ax = train.plot()
test.plot(ax=ax)
plt.legend(['train', 'test'])
plt.show()

# 将训练集和测试集缩放为[-1,1]
# 缩放公式为：xi = (xi - mean(xi))/(max(xi)-min(xi))
# 后续可以利用此公式进行数据还原
# 进行数据缩放主要为了方便模型训练
scaler = MinMaxScaler(feature_range=(-1, 1))
train_sc = scaler.fit_transform(train)
test_sc = scaler.transform(test)
# 训练集取除去最后一个元素的所有值
X_train = train_sc[:-1]
# 训练集取去除第一个元素的所有值
y_train = train_sc[1:]
X_test = test_sc[:-1]
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
# 用train集合进行预测
y_train_pred_nn = nn_model.predict(X_train)
print("The R2 score on the Train set is:\t{:0.3f}".format(r2_score(y_train, y_train_pred_nn)))
print("The R2 score on the Test set is:\t{:0.3f}".format(r2_score(y_test, y_pred_test_nn)))
# 保存训练好的模型
dirs = 'midModel'
if not os.path.exists(dirs):
    os.makedirs(dirs)
joblib.dump(nn_model, dirs+'/ANN.pkl')
# 通过下面方式进行模型调用
# ANN = joblib.loab(dirs + '/ANN.pkl')


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
history_lstm_model = lstm_model.fit(X_train_lstm, y_train, epochs=100, batch_size=1, verbose=1, shuffle=False,
                                    callbacks=[early_stop])
# 预测值
# 用test集合进行预测
y_pred_test_lstm = lstm_model.predict(X_test_lstm)
# 用train集合进行预测
y_train_pred_lstm = lstm_model.predict(X_train_lstm)
print("The R2 score on the Train set is:\t{:0.3f}".format(r2_score(y_train, y_train_pred_lstm)))
print("The R2 score on the Test set is:\t{:0.3f}".format(r2_score(y_test, y_pred_test_lstm)))
# 模型保存
joblib.dump(lstm_model, dirs+'/LSTM.pkl')

# 两种模型MSE比较
# model.evaluate输入数据(data)和金标准(label),然后将预测结果与金标准相比较,得到两者误差并输出.
# model.predict输入数据(data),输出预测结果
nn_test_mse = nn_model.evaluate(X_test, y_test, batch_size=1)
lstm_test_mse = lstm_model.evaluate(X_test_lstm, y_test, batch_size=1)
print('ANN: %f' % nn_test_mse)
print('LSTM: %f' % lstm_test_mse)

# 进行预测
nn_y_pred_test = nn_model.predict(X_test)
# lstm_y_pred_test = lstm_model.predict(X_test)
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='True')
plt.plot(y_pred_test_nn, label='NN')
plt.title("ANN's Prediction")
plt.xlabel('Observation')
plt.ylabel('Value')
plt.legend()
plt.show()

# LSTM预测
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='True')
plt.plot(y_pred_test_lstm, label='LSTM')
plt.title("LSTM's Prediction")
plt.xlabel('Observation')
plt.ylabel('Value')
plt.legend()
plt.show()
