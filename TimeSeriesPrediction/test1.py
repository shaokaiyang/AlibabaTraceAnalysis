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
split_boundary = '2018-11-07 0:00:00'
filename = 'task_submit_currency_second_all.csv'
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
plt.savefig('currency_second1.png',bbox_inches='tight')
plt.show()