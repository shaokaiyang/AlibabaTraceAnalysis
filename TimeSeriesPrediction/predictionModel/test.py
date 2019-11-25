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
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

time_field = 'Date'
value_field = 'Value'
# split_boundary = '2018-11-06 0:00:00'
# 2017年数据建模分界点
split_boundary = '2017-11-01 1:55:00'
# filename = 'currency_minute_normalized.csv'
# 2017年数据文件
filename1 = 'task_submit_interval_normalized.csv'
filename2 = 'task_submit_number_normalized.csv'
filename3 = 'task_submit_time2017.csv'
dirs = 'midModel'

# 将数据读为一个数据框架
df1 = pd.read_csv(filename1)
df2 = pd.read_csv(filename2)
df3 = pd.read_csv(filename3)
# 将时间列改为标准时间格式
df1[time_field] = pd.to_datetime(df1[time_field])
# 将索引设置为时间字段
df1 = df1.set_index([time_field], drop=True)
df2[time_field] = pd.to_datetime(df2[time_field])
# 将索引设置为时间字段
df2 = df2.set_index([time_field], drop=True)


# 按2018-11-06拆分训练集和测试集
split_date = pd.Timestamp(split_boundary)
# 划定测试集和训练集,以分割时间为边界
train1 = df1.loc[:split_date]
test1 = df1.loc[split_date:]

train2 = df2.loc[:split_date]
test2 = df2.loc[split_date:]

# # 训练集和测试集表示在图上
# plt.figure(figsize=(15, 8))
# plt.plot(train1['Value'], label='Train')
# plt.plot(test1['Value'], label='Test')
# plt.xlabel('Time')
# plt.ylabel('Currency')
# plt.savefig('currency_minute_raw.jpg', bbox_inches='tight')
# plt.show()

# # 训练集和测试集表示在图上
# plt.figure(figsize=(15, 8))
# plt.plot(train2['Value'], label='Train')
# plt.plot(test2['Value'], label='Test')
# plt.xlabel('Time')
# plt.ylabel('Currency')
# plt.savefig('currency_minute_raw.jpg', bbox_inches='tight')
# plt.show()
#
# 训练集和测试集表示在图上
plt.figure(figsize=(20, 8))
plt.plot(df3, label='Train')
x = range(0, 86401, 3600)
y = ('0h', '1h', '2h', '3h', '4h', '5h', '6h', '7h', '8h', '9h', '10h', '11h', '12h', '13h', '14h', '15h',
     '16h', '17h', '18h', '19h', '20h', '21h', '22h', '23h', '24h')
plt.xlabel('Time', fontname='Arial', )
plt.ylabel('Currency')
plt.xticks(x, y)
plt.savefig('currency_minute_raw1.jpg', bbox_inches='tight')
plt.show()

# 移动平均图
def draw_trend(timeseries, size):
    f = plt.figure(facecolor='white')
    # 对size个数据进行移动平均
    rol_mean = timeseries.rolling(window=size).mean()
    # 对size个数据移动平均的方差
    rol_std = timeseries.rolling(window=size).std()

    timeseries.plot(color='blue', label='Original')
    rol_mean.plot(color='red', label='Rolling Mean')
    rol_std.plot(color='black', label='Rolling standard deviation')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()




