# -*- coding: UTF-8 -*-
# 提取三天(3,4,5天)中task的开始时间（可以认为是提交时间）
# 处理后的数据中包含任务类型信息和提交时间

import pandas as pd
import numpy as np
import time
import _thread


# 输入输出路径前缀
file_input_path_prefix = 'D:\\AlibabaTraceAnalysis\\SplitResult\\batch_task_'
file_output_path_prefix = 'D:\\AlibabaTraceAnalysis\\AnalysisResult\\task_start_time_'

# 输入路径
file_input_path0 = 'D:\\AlibabaTraceAnalysis\\SplitResult\\batch_task_0.csv'
file_input_path1 = 'D:\\AlibabaTraceAnalysis\\SplitResult\\batch_task_1.csv'
file_input_path2 = 'D:\\AlibabaTraceAnalysis\\SplitResult\\batch_task_2.csv'
# 记录输入路径的列表
file_input_path_list = [file_input_path0, file_input_path1, file_input_path2]
# 输出路径前缀
file_output_path0 = 'D:\\AlibabaTraceAnalysis\\AnalysisResult\\task_start_time_0.csv'
file_output_path1 = 'D:\\AlibabaTraceAnalysis\\AnalysisResult\\task_start_time_1.csv'
file_output_path2 = 'D:\\AlibabaTraceAnalysis\\AnalysisResult\\task_start_time_2.csv'
file_output_path_list = [file_output_path0, file_output_path1, file_output_path2]

# 分析的天数
days = 8

if __name__ == '__main__':
    # 统计资源信息
    start = time.time()
    count = 0
    for i in range(days):
        file_path_in = file_input_path_prefix + str(i) + '.csv'
        file_path_out = file_output_path_prefix + str(i) + '.csv'
        reader = pd.read_csv(file_path_in, header=None, usecols=[3, 5], chunksize=10000000)
        for chunk in reader:
            count += chunk.shape[0]
            chunk.to_csv(file_path_out, index=False, header=False, mode='a')
            print("第 %s 个文件的前 %s 行，处理完毕" % (i, count))
        count = 0
    end = time.time()
    print('Running time: %s Seconds' % (end - start))
















