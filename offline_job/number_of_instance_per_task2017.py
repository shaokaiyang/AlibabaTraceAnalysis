# -*- coding: UTF-8 -*-
# 分析2017年trace数据
# 输入为：batch_task文件
# 输出为：每个task所包含的instance数量

import pandas as pd
import numpy as np
import time
import _thread


# 输入路径
file_input_path = 'D:\\AlibabaTraceAnalysis\\alibaba_clustertrace_2017\\batch_task.csv'
# 输出路径前缀
file_output_path = 'D:\\AlibabaTraceAnalysis\\AnalysisResult\\number_of_instance_per_task2017.csv'


if __name__ == '__main__':
    # 统计资源信息
    start = time.time()
    count = 0

    # 读取task_start_time中的提交时间
    reader = pd.read_csv(file_input_path, header=None, usecols=[4])
    reader.to_csv(file_output_path, index=False, header=False, mode='a')

    end = time.time()
    print('Running time: %s Seconds' % (end - start))
















