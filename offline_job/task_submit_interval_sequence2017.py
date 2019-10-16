# -*- coding: UTF-8 -*-
# 分析2017年trace数据
# 输入为：所有任务提交时间间隔
# 输出为：去重后的任务提交时间间隔，这样数量与并发度相对应

import pandas as pd
import numpy as np
import time
import _thread


# 输入路径
file_input_path = 'D:\\AlibabaTraceAnalysis\\AnalysisResult\\task_submit_interval_all2017.csv'
# 输出路径前缀
file_output_path0 = 'D:\\AlibabaTraceAnalysis\\AnalysisResult\\task_submit_interval2017.csv'
file_output_path1 = 'D:\\AlibabaTraceAnalysis\\AnalysisResult\\task_submit_number2017.csv'


if __name__ == '__main__':
    # 统计资源信息
    start = time.time()
    count = 0

    # 读取task_start_time中的提交时间
    reader = pd.read_csv(file_input_path, header=None, usecols=[0])
    # 将读取到的提交时间转化为列表进行处理
    result_list = reader.values.tolist()
    print(result_list)
    print('提交的task数量为 %s' % len(result_list))
    interval_list = []
    number_list = []
    pre_interval = result_list[0][0]
    interval_list.append(pre_interval)
    count = 0
    for k in result_list:
        j = k[0]
        if j != pre_interval:
            pre_interval = j
            interval_list.append(pre_interval)
            number_list.append(count)
            count = 1
        else:
            count = count + 1
    print(count)
    print(len(interval_list))
    print(len(number_list))
    save1 = pd.DataFrame({'task_submit_interval2017': interval_list})
    save1.to_csv(file_output_path0, index=False, header=False, mode='a')
    save2 = pd.DataFrame({'task_submit_number2017': number_list})
    save2.to_csv(file_output_path1, index=False, header=False, mode='a')

    end = time.time()
    print('Running time: %s Seconds' % (end - start))
















