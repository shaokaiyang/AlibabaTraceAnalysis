import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from collections import namedtuple


def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2. - 0.1, 1.01 * height, '%s' % int(height), fontsize=15)
       # plt.text(rect.get_x() + rect.get_width() / 2. - 0.1, 1.01 * height, '%.1f' % height, fontsize=15)
n_groups = 6

#1G
average = (37, 42, 98, 75, 96, 81)
#10g
#min = ( 65.3, 63.69, 96.3)

#max = ( 117,128,159,184)

fig, ax = plt.subplots()

index = np.arange(n_groups)
bar_width = 0.5

opacity = 0.5
error_config = {'ecolor': '0.3'}

rects1 = ax.bar(index+bar_width, average, bar_width,
                alpha=opacity, fill='false',
                edgecolor='black', color='white',
                linewidth='1.5',
                 align='center', hatch='//')

# rects2 = ax.bar(index + bar_width, min, bar_width,
#                 alpha=opacity, fill='false',
#                 edgecolor='black', color='white',
#                 linewidth='1.5',
#                 label='10G', align='center', hatch='xx')
# rects3 = ax.bar(index + bar_width * 2, max, bar_width,
#                 alpha=opacity, fill='false',
#                 edgecolor='black', color='white',
#                 linewidth='1.5',
#                 label='Maximum', align='center', hatch='//')

autolabel(rects1)
#autolabel(rects2)
#autolabel(rects3)

ax.set_ylabel('Complete Time(s)', fontsize=13, fontname='Arial')
# ax.set_title('Scores by group and gender')
ax.set_xticks(index + bar_width / 1.1)
plt.tick_params(labelsize=13)
ax.set_xticklabels(('PI', 'WordCount', 'Terasort', 'PageRank', 'K-means','NaiveBayes'),fontname='Arial')
ax.legend(loc='upper left', fontsize=13, frameon='false')

fig.tight_layout()
fig.savefig('loadbar.jpg',bbox_inches='tight')
plt.show()
