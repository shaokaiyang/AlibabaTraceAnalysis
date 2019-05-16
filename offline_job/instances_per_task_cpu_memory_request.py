# -*- coding: UTF-8 -*-
# 分析2018年 batch-task文件中的信息，此文件提取每个task包含的instance数量，以及每个task申请的资源量
# 输入数据为：batch_task012
# 输出数据为: number_of_instacne_per_task  CPU_memory_request_per_instance

import pandas as pd
import numpy as np
import time
import _thread


# 输入路径
file_input_path0 = 'D:\\AlibabaTraceAnalysis\\SplitResult\\batch_task_0.csv'
file_input_path1 = 'D:\\AlibabaTraceAnalysis\\SplitResult\\batch_task_1.csv'
file_input_path2 = 'D:\\AlibabaTraceAnalysis\\SplitResult\\batch_task_2.csv'
# 记录输入路径的列表
file_input_path_list = [file_input_path0, file_input_path1, file_input_path2]
# 输出路径前缀，用于记录task所包含的instance数量
file_output_path0 = 'D:\\AlibabaTraceAnalysis\\AnalysisResult\\number_of_instance_per_task_0.csv'
file_output_path1 = 'D:\\AlibabaTraceAnalysis\\AnalysisResult\\number_of_instance_per_task_1.csv'
file_output_path2 = 'D:\\AlibabaTraceAnalysis\\AnalysisResult\\number_of_instance_per_task_2.csv'
file_output_path_list1 = [file_output_path0, file_output_path1, file_output_path2]
# 输出路径前缀，用于记录task申请的资源量
file_output0 = 'D:\\AlibabaTraceAnalysis\\AnalysisResult\\CPU_memory_request_per_instance_0.csv'
file_output1 = 'D:\\AlibabaTraceAnalysis\\AnalysisResult\\CPU_memory_request_per_instance_1.csv'
file_output2 = 'D:\\AlibabaTraceAnalysis\\AnalysisResult\\CPU_memory_request_per_instance_2.csv'
file_output_path_list2 = [file_output0, file_output1, file_output2]
# 分析的天数
days = 3

if __name__ == '__main__':
    # 统计资源信息
    start = time.time()
    count = 0
    for i in range(days):
        # 记录资源申请情况
        reader1 = pd.read_csv(file_input_path_list[i], header=None, usecols=[7, 8])
        reader1.to_csv(file_output_path_list2[i], index=False, header=False, mode='a')
        # 记录每个task所包含的instance数量
        reader2 = pd.read_csv(file_input_path_list[i], header=None, usecols=[1])
        reader2.to_csv(file_output_path_list1[i], index=False, header=False, mode='a')

    end = time.time()
    print('Running time: %s Seconds' % (end - start))
















