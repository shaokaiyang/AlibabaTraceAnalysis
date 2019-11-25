# -*- coding: UTF-8 -*-
# 分析2017年trace数据
# 输入为：所有任务提交时间间隔
# 输出为：任务提交时间点序列

import pandas as pd
import numpy as np
import time
import _thread


# 输入路径
file_input_path = 'D:\\AlibabaTraceAnalysis\\AnalysisResult\\task_submit_interval_all2017.csv'
# 输出路径前缀
file_output_path0 = 'D:\\AlibabaTraceAnalysis\\AnalysisResult\\task_submit_time2017.csv'


if __name__ == '__main__':
    # 统计资源信息
    start = time.time()
    count = 0

    # 读取task_start_time中的提交时间
    reader = pd.read_csv(file_input_path, header=None, usecols=[0])
    # 将读取到的提交时间转化为列表进行处理
    result_list = reader.values.tolist()
    number_list = [0]*57350
    pre_interval = result_list[0][0]
    sum_time = 0
    count = 0
    for k in result_list:
        j = k[0]
        if j != pre_interval:
            pre_interval = j
            number_list[sum_time] = count
            sum_time = sum_time + pre_interval
            count = 1
        else:
            count = count + 1
    save1 = pd.DataFrame({'task_submit_time2017': number_list})
    save1.to_csv(file_output_path0, index=False, header=False, mode='a')
    end = time.time()
    print('Running time: %s Seconds' % (end - start))
















