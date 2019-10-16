# -*- coding: UTF-8 -*-
# 分析2017年trace数据
# 输入为：batch_task文件
# 输出为：所有任务提交的时间间隔，相同提交时间点提交的任务数量

import pandas as pd
import numpy as np
import time
import _thread


# 输入路径
file_input_path = 'D:\\AlibabaTraceAnalysis\\alibaba_clustertrace_2017\\batch_task.csv'
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
    result_list = reader.sort_values(by=[0]).values.tolist()
    print(result_list)
    print('提交的task数量为 %s' % len(result_list))
    interval_list = []
    number_list = []
    pre_time = result_list[0][0]
    last_time = result_list[0][0]
    count = 0
    for k in result_list:
        j = k[0]
        if j == last_time:
            count += 1
            interval_list.append(j - pre_time)
        else:
            number_list.append(count)
            pre_time = last_time
            last_time = j
            count = 1
            interval_list.append(j - pre_time)
    number_list.append(count)
    print('存在 %s 个时间间隔' % len(interval_list))
    print('存在 %s 个提交时间点' % len(number_list))
    save1 = pd.DataFrame({'task_submit_interval2017': interval_list})
    save1.to_csv(file_output_path0, index=False, header=False, mode='a')
    save2 = pd.DataFrame({'task_submit_number2017': number_list})
    save2.to_csv(file_output_path1,index=False, header=False, mode='a')

    end = time.time()
    print('Running time: %s Seconds' % (end - start))
















