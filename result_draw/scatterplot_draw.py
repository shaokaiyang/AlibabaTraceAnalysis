# -*- coding: UTF-8 -*-
# 绘制散点图，输入为DataFrame
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# 用来正常显示中文标签
plt.rcParams['font.sans-serif'] = ['SimHei']
# 用来正常显示负号
# 有中文出现的情况，需要u'内容'
plt.rcParams['axes.unicode_minus'] = False


file_input_path = 'D:\\AlibabaTraceAnalysis\\AnalysisResult\\task_start_time_0.csv'
reader = pd.read_csv(file_input_path, header=None, usecols=[1])
y = reader[0:1000].drop_duplicates().values
# y = np.random.rand(1000)
# print(y)
x = list(range(0, len(y)))
print(len(y))
plt.scatter(x, y, s=0.5)
plt.show()

