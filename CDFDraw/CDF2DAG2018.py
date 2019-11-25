# -*- coding: UTF-8 -*-
# 绘制cdf图，输入为DataFrame

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
import statsmodels.api as sm

# 绘制两组数据的CDF图像在一个图片里
# 读取数据文件,并将数据文件转化为数组
filename = 'DAGNumber.csv'
with open(filename, 'r') as csvFile:
    # 将CSV文件转化为矩阵
    matrix = np.loadtxt(csvFile, delimiter=",", skiprows=0)
    # 将每一列数据存储在数组中
    array1 = matrix[:, 0]
    array2 = matrix[:, 1]

    # 转换数据类型
    Alone = array1.astype(int)
    Mixed = array2.astype(int)


# 使用ECDF函数画CDF曲线
ecdfAlone = sm.distributions.ECDF(Alone)
x1 = np.linspace(min(Alone), max(Alone), 100)
y1 = ecdfAlone(x1)
ecdfMixed = sm.distributions.ECDF(Mixed)
x2 = np.linspace(min(Mixed), max(Mixed), 100)
y2 = ecdfMixed(x2)

line1, = plt.plot(x1, y1, 'r-', label='First_Day_DAG_Number', linewidth=1)
line2, = plt.plot(x2, y2, 'b--', label='Second_Day_DAG_Number', linewidth=1)

plt.legend(bbox_to_anchor=(0.65, 0.3), loc='best', borderaxespad=0., frameon=False)
plt.grid(which='major', axis='y', linestyle='--', linewidth=1)
plt.xlabel('DAG Numver')
plt.savefig('2018dagnumver_cdf.jpg', bbox_inches='tight')
# plt.axvline(1500, -1, 2, color='r', label='nim')
plt.show()
