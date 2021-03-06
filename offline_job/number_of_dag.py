# -*- coding: UTF-8 -*-
# 分析2018年 batch-task文件中的信息，提取task的运行时间
# 输入数据为：batch_task012
# 输出数据为: task运行时间

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
file_output_path0 = 'D:\\AlibabaTraceAnalysis\\AnalysisResult\\number_of_dag_0.csv'
file_output_path1 = 'D:\\AlibabaTraceAnalysis\\AnalysisResult\\number_of_dag_1.csv'
file_output_path2 = 'D:\\AlibabaTraceAnalysis\\AnalysisResult\\number_of_dag_2.csv'
file_output_path_list1 = [file_output_path0, file_output_path1, file_output_path2]

# 分析的天数
days = 3

if __name__ == '__main__':
    # 统计资源信息
    start = time.time()
    for i in range(days):
        # 记录资源申请情况
        reader = pd.read_csv(file_input_path_list[i], header=None, usecols=[0])
        result_list = reader.values.tolist()
        print(result_list)
        print('提交的task数量为 %s' % len(result_list))
        number_list = []
        for k in result_list:
            j = k[0].split('_')
            number_list.append(len(j) - 1)

        print('存在 %s 个job' % len(number_list))
        save1 = pd.DataFrame({'number_of_dah': number_list})
        save1.to_csv(file_output_path_list1[i], index=False, header=False, mode='a')

    end = time.time()
    print('Running time: %s Seconds' % (end - start))
















