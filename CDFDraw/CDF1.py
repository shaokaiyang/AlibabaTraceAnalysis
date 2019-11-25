# -*- coding: UTF-8 -*-
# 绘制cdf图，输入为DataFrame

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
import statsmodels.api as sm

# 绘制两组数据的CDF图像在一个图片里
# 读取数据文件,并将数据文件转化为数组
df = pd.read_csv('onlineapp_include_containers.csv', delimiter=",", skiprows=0)
Alone = np.array(df.values[:, 0])

# 使用ECDF函数画CDF曲线
ecdfAlone = sm.distributions.ECDF(Alone)
x1 = np.linspace(min(Alone), max(Alone), 100)
y1 = ecdfAlone(x1)

line1, = plt.plot(x1, y1, label='Container_Per_Online_Application', linewidth=1)

plt.legend(bbox_to_anchor=(0.65, 0.3), loc='upper right', borderaxespad=0., frameon=False)
plt.grid(which='major', axis='y', linestyle='--', linewidth=1)
plt.xlabel('Container Per Online Application')
# plt.axvline(1500, -1, 2, color='r', label='nim')
# plt.savefig('containerperonlineapp_cdf.jpg', bbox_inches='tight')
plt.show()

