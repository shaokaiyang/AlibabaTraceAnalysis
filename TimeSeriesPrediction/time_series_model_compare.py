# -*- coding: UTF-8 -*-
# 通过移动平均、K-近邻、自动ARIMA、先知（Prophet）、LSTM等方法进行时间序列分析，分析股市的收盘价

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from pandas.plotting import register_matplotlib_converters


scaler = MinMaxScaler(feature_range=(0, 1))

# read the file
df = pd.read_csv('stock_data.csv')

# setting index as date
df['Date'] = pd.to_datetime(df.Date, format='%Y-%m-%d')
df.index = df['Date']
data = df.sort_index(ascending=True, axis=0)
print(df.head())
# 画原始数据图像
# plt.figure(figsize=(10, 6))
# df['Close'].plot()
# plt.show()


# 移动平均
# creating dataframe with date and the target variable
new_data_mean = pd.DataFrame(index=range(0, len(df)), columns=['Date', 'Close'])
# 数据预处理
for i in range(0, len(data)):
    new_data_mean['Date'][i] = data['Date'][i]
    new_data_mean['Close'][i] = data['Close'][i]
# splitting into train and validation
train_mean = new_data_mean[:1734]
valid_mean = new_data_mean[1734:]
print(new_data_mean.shape, train_mean.shape, valid_mean.shape)
print(train_mean['Date'].min(), train_mean['Date'].max(), valid_mean['Date'].min(), valid_mean['Date'].max())
# 创建预测集
preds = []
# 生成测试集大小的预测集
for i in range(0, 278):
    a = train_mean['Close'][len(train_mean)-278+i:].sum() + sum(preds)
    b = a/278
    preds.append(b)
# calculate rmse
rms_mean = np.sqrt(np.mean(np.power((np.array(valid_mean['Close'])-preds), 2)))
print(rms_mean)
# plot
valid_mean.insert(2, 'Predictions', preds)
plt.plot(train_mean['Close'])
plt.plot(valid_mean[['Close', 'Predictions']])
plt.show()

# KNN
from sklearn import neighbors
from sklearn.model_selection import GridSearchCV
x_train_knn = train_mean.drop('Close', axis=1)
y_train_knn = train_mean['Close']
x_valid_knn = valid_mean.drop('Close', axis=1)
y_valid_knn = valid_mean['Close']

# scaling data
x_train_scaled = scaler.fit_transform(x_train_knn)
x_train_knn = pd.DataFrame(x_train_scaled)
x_valid_scaled = scaler.fit_transform(x_valid_knn)
x_valid_knn = pd.DataFrame(x_valid_scaled)
# using gridsearch to find the best parameter
params = {'n_neighbors': [2, 3, 4, 5, 6, 7, 8, 9]}
knn = neighbors.KNeighborsRegressor()
model = GridSearchCV(knn, params, cv=5)
# fit the model and make predictions
model.fit(x_train_knn, y_train_knn)
preds = model.predict(x_valid_knn)
# rmse
rms_knn = np.sqrt(np.mean(np.power((np.array(y_valid_knn)-np.array(preds)), 2)))
print(rms_knn)
# plot
# plot
valid_mean.insert(3, 'Predictions_knn', preds)
plt.plot(train_mean['Close'])
plt.plot(valid_mean[['Close', 'Predictions_knn']])
plt.show()



