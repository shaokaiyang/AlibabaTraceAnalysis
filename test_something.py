
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import scipy.stats as stats
import pandas as pd

s = 'R2_1'
m = s.split('_')
print(m)



# data = [0.8, 11.7, 2.8, 11.9, 6.1, 1, 34.8, 3.8, 5.2, 15.0, 10.3, 12.3,
#         8.2, 0.6, 1.7, 14.5, 8.3, 28.9, 3.1, 7.3, 10.2, 8.9, 0.1, 15.5, 5.7, 0.7, 8.3, 0.9, 40.7, 2.9]
# m = np.mean(data)
# n = np.median(data)
# print(np.mean(data))
# print(np.median(data))
# print(np.var(data))
# print(np.max(data))
# print(np.min(data))
# print(stats.kstest(data, ''))
# plt.hist(data)
# # plt.xlim(0, 50)
# # plt.ylim(0, 5)
# plt.show()
# plt.figure(figsize=(16, 10), dpi=80)
# sns.kdeplot(data, shade=True, color="g", label="x", alpha=.7)
# plt.legend()
# plt.show()
# df = pd.DataFrame({'co1': [2, 1, 8, 5, ], 'co2': []})
# print(df.sort_values(by=['co1']))
# result_list = [1, 1, 3, 5, 5, 8]
# interval_list = []
# pre_time = 0
# last_time = result_list[0]
# count = 0
# for j in result_list:
#         if j == last_time:
#                 count += 1
#         else:
#                 interval_list.append(count)
#                 last_time = j
#                 count = 1
# interval_list.append(count)
# print(interval_list)
