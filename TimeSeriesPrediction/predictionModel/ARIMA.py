# -*- coding: UTF-8 -*-
# 使用LSTM进行18年并发度数据建模预测

import os
import warnings
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
# 判断数据的稳定性
from statsmodels.tsa.stattools import adfuller
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import itertools


# 移动平均图
def draw_trend(timeSeries, size):
    f = plt.figure(facecolor='white')
    # 对size个数据进行移动平均
    rol_mean = timeSeries.rolling(window=size).mean()
    # 对size个数据进行加权移动平均
    rol_weighted_mean = pd.ewma(timeSeries, span=size)

    timeSeries.plot(color='blue', label='Original')
    rol_mean.plot(color='red', label='Rolling Mean')
    rol_weighted_mean.plot(color='black', label='Weighted Rolling Mean')
    plt.legend(loc='best')
    plt.title('Rolling Mean')
    plt.show()


def draw_ts(timeSeries):
    f = plt.figure(facecolor='white')
    timeSeries.plot(color='blue')
    plt.show()


'''
　　ADF检验，p值小于0.05，则为稳定数据
'''


def testStationarity(ts):
    dftest = adfuller(ts)
    # 对上述函数求得的值进行语义描述
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    return dfoutput


# 自相关和偏相关图，默认阶数为31阶
def draw_acf_pacf(ts, lags=50):
    f = plt.figure(facecolor='white')
    ax1 = f.add_subplot(211)
    plot_acf(ts, lags=50, ax=ax1)
    ax2 = f.add_subplot(212)
    plot_pacf(ts, lags=50, ax=ax2)
    plt.show()

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

df = df['Value']

# print(testStationarity(df))
# draw_acf_pacf(df)

# # Define the p, d and q parameters to take any value between 0 and 2
# p = d = q = range(0, 2)
# # Generate all different combinations of p, q and q triplets
# pdq = list(itertools.product(p, d, q))
# # Generate all different combinations of seasonal p, q and q triplets
# seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
# warnings.filterwarnings("ignore") # specify to ignore warning messages
# for param in pdq:
#     for param_seasonal in seasonal_pdq:
#         try:
#             mod = sm.tsa.statespace.SARIMAX(train,
#                                             order=param,
#                                             seasonal_order=param_seasonal,
#                                             enforce_stationarity=False,
#                                             enforce_invertibility=False)
#
#             results = mod.fit()
#
#             print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
#         except:
#             continue


currency_moving_avg = test.copy()
import statsmodels.api as sm

mod = sm.tsa.statespace.SARIMAX(df, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12),
                                enforce_stationarity=False, enforce_invertibility=False)

results = mod.fit()
results.summary()
results.plot_diagnostics(figsize=(15,12))
plt.savefig('ARIMA_test.jpg', bbox_inches='tight')
plt.show()

pred = results.get_prediction(start=pd.to_datetime("2018-11-06 0:00:00"), dynamic=False)
ax = df["2018-11-06 0:00:00":].plot(label='True')
pred.predicted_mean.plot(ax = ax, label='ARIMA', alpha=.7,figsize=(15, 8))
np.savetxt('ARIMA.csv', pred.predicted_mean, delimiter=',')

ax.set_xlabel('Date', fontname='Arial',fontsize=22)
ax.set_ylabel('Concurrency',fontname='Arial',fontsize=22)
plt.legend(loc='upper right',fontsize=22)
plt.tick_params(labelsize=18)
plt.savefig('ARIMA.jpg', bbox_inches='tight')
plt.show()

# currency_moving_avg['Value'] = fit1.predict(start='2018-11-06', end='2018-11-08', dynamic=True)
# currency_moving_avg.to_csv('ARIMA_prediction.csv', index=0)
# plt.figure(figsize=(15, 8))
# plt.plot(train['Value'], label='Train')
# plt.plot(test['Value'], label='Test')
# plt.plot(currency_moving_avg['Value'], label='Simple_avg')
# plt.xlabel('Time')
# plt.ylabel('Currency')
# plt.legend(loc='best')
# plt.savefig('ARIMA.jpg', bbox_inches='tight')
# plt.show()
#
R2 = r2_score(test.Value, pred.predicted_mean)
MSE = mean_squared_error(test.Value, pred.predicted_mean)
RMSE = math.sqrt(MSE)
MeanAE = mean_absolute_error(test.Value,pred.predicted_mean)
MedianAE = median_absolute_error(test.Value, pred.predicted_mean)
print("R2: %s" % R2)
print("MSE: %s" % MSE)
print("RMSE: %s" % RMSE)
print("MeanAE: %s" % MeanAE)
print("MedianAE: %s" % MedianAE)


