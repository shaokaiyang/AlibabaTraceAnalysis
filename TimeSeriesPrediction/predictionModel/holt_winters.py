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
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing,Holt
import statsmodels.api as sm

time_field = 'Date'
value_field = 'Value'
split_boundary = '2018-11-06 0:00:00'
# 2017年数据建模分界点
# split_boundary = '2017-11-01 1:55:00'
filename = 'currency_minute_normalized.csv'
# 2017年数据文件
# filename = 'task_submit_interval_normalized.csv'
# filename = 'task_submit_number_normalized.csv'
dirs = 'midModel'

# 将数据读为一个数据框架
df = pd.read_csv(filename)
# 将时间列改为标准时间格式
df[time_field] = pd.to_datetime(df[time_field])
# 将索引设置为时间字段
df = df.set_index([time_field], drop=True)

# 按2018-11-06拆分训练集和测试集
split_date = pd.Timestamp(split_boundary)
# 划定测试集和训练集,以分割时间为边界
train = df.loc[:split_date]
test = df.loc[split_date:]
# 训练集和测试集表示在图上
# plt.figure(figsize=(15, 8))
# plt.plot(train['Value'], label='Train')
# plt.plot(test['Value'], label='Test')
# plt.xlabel('Time')
# plt.ylabel('Currency')
# plt.savefig('currency_minute_raw.jpg', bbox_inches='tight')
# plt.show()


currency_moving_avg = test.copy()
fit1 = ExponentialSmoothing(np.array(train['Value']), seasonal_periods=1440, trend='add', seasonal='add').fit()
currency_moving_avg['Value'] = fit1.forecast(len(test))
currency_moving_avg.to_csv('holt_winter_prediction.csv', index=0)
plt.figure(figsize=(15, 8))
plt.plot(train['Value'], label='Train')
plt.plot(test['Value'], label='Test')
plt.plot(currency_moving_avg['Value'], label='Simple_avg')
plt.xlabel('Time')
plt.ylabel('Currency')
plt.legend(loc='best')
plt.savefig('holt_winter.jpg', bbox_inches='tight')
plt.show()

R2 = r2_score(test.Value, currency_moving_avg.Value)
MSE = mean_squared_error(test.Value, currency_moving_avg.Value)
RMSE = math.sqrt(MSE)
MeanAE = mean_absolute_error(test.Value,currency_moving_avg.Value)
MedianAE = median_absolute_error(test.Value, currency_moving_avg.Value)
print("R2: %s" % R2)
print("MSE: %s" % MSE)
print("RMSE: %s" % RMSE)
print("MeanAE: %s" % MeanAE)
print("MedianAE: %s" % MedianAE)


