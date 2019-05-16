# -*- coding: UTF-8 -*-
# 提取三天(4,5,6天)中task的开始时间（可以认为是提交时间）
# 输入数据为：任务的提交时间
# 输出数据为: 分析得到提交的时间间隔为1秒，所以保存的数据为每秒提交的任务数量

import pandas as pd
import numpy as np
import time
import _thread


# 输入路径
file_input_path0 = 'D:\\AlibabaTraceAnalysis\\AnalysisResult\\task_start_time_0.csv'
file_input_path1 = 'D:\\AlibabaTraceAnalysis\\AnalysisResult\\task_start_time_1.csv'
file_input_path2 = 'D:\\AlibabaTraceAnalysis\\AnalysisResult\\task_start_time_2.csv'
# 记录输入路径的列表
file_input_path_list = [file_input_path0, file_input_path1, file_input_path2]
# 输出路径前缀
file_output_path0 = 'D:\\AlibabaTraceAnalysis\\AnalysisResult\\task_submit_interval_0.csv'
file_output_path1 = 'D:\\AlibabaTraceAnalysis\\AnalysisResult\\task_submit_interval_1.csv'
file_output_path2 = 'D:\\AlibabaTraceAnalysis\\AnalysisResult\\task_submit_interval_2.csv'
file_output_path_list = [file_output_path0, file_output_path1, file_output_path2]
# 分析的天数
days = 3
day = 86400

if __name__ == '__main__':
    # 统计资源信息
    start = time.time()
    count = 0
    for i in range(days):
        # 读取task_start_time中的提交时间
        reader = pd.read_csv(file_input_path_list[i], header=None, usecols=[1])
        result = reader.sort_values(by=[1])
        result_list = result.values.tolist()
        interval_list = []
        last_time = result_list[0][0]
        count = 0
        for k in result_list:
            j = k[0]
            if j == last_time:
                count += 1
            else:
                interval_list.append(count)
                last_time = j
                count = 1
        interval_list.append(count)
        print(len(interval_list))
        # save = pd.DataFrame({'task_submit_interval': interval_list})
        # save.to_csv(file_output_path_list[i], index=False, header=False, mode='a')
        # print(len(result_list))

    end = time.time()
    print('Running time: %s Seconds' % (end - start))
















