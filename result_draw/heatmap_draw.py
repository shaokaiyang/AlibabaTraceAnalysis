# -*- coding: UTF-8 -*-
# 绘制热图，输入为DataFrame

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set()
np.random.seed(0)
uniform_data = np.random.rand(10, 12)
ax = sns.heatmap(uniform_data, vmin=0, vmax=1)
plt.show()

