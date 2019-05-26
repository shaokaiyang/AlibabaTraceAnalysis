# -*- coding: UTF-8 -*-
# 记录matplotlib相关用法

# import matplotlib.pyplot as plt
# import numpy as np
#
#
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# sns.set()
# np.random.seed(0)
# uniform_data = []
# print(uniform_data)
# ax = sns.heatmap(uniform_data)
# plt.show()

import plotly.plotly as py
import plotly.graph_objs as go

import numpy as np

y0 = np.random.randn(50) - 1
y1 = np.random.randn(50) + 1
# 生成两个随机集合

trace0 = go.Box(
    y=y0
)
trace1 = go.Box(
    y=y1
)
data = [trace0, trace1]
print(data)
py.iplot(data)
