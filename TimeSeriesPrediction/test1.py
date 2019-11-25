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
filename = 'task_submit_currency_minute_all.csv'

# 将数据读为一个数据框架
df = pd.read_csv(filename)
# 将时间列改为标准时间格式
df[time_field] = pd.to_datetime(df[time_field])
# 将索引设置为时间字段
df = df.set_index([time_field], drop=True)


plt.figure(figsize=(15, 8))
plt.plot(df['Value'])
plt.xlabel('Date', fontname='Arial', fontsize=22)
plt.ylabel('Concurrency', fontname='Arial', fontsize=22)
plt.tick_params(labelsize=18)
plt.savefig('currency_minite_raw.jpg', bbox_inches='tight')
plt.show()


