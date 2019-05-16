## Pandas 常见操作
### DataFrame 常见属性信息查看
- df.head(5)  显示前5行内容
- df.tail(5)  显示后5行内容
- df.info()  显示DataFrame的概览信息，索引、行列属性等
- df.describe()  显示DataFrame的统计信息，条目数、平均值、方差、最小值、最大值、不同分位点值
### DataFrame 对某一列进行排序
df.sort_values(by=['col'], ascending=true, na_position=first)  by后面跟随需要排序的列可以根据列名或者列标号，ascending表示升序还是降序，
na_position表示NAN位于什么地方