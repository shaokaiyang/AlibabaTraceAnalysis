# -*- coding: UTF-8 -*-
# 分析2017年trace数据
# 输入为：batch_task文件
# 输出为：每个task实际执行时间

import pandas as pd
import numpy as np
import time
import _thread


# 输入路径
file_input_path = 'D:\\AlibabaTraceAnalysis\\alibaba_clustertrace_2017\\batch_task.csv'
# 输出路径前缀
file_output_path = 'D:\\AlibabaTraceAnalysis\\AnalysisResult\\task_duration2017.csv'


if __name__ == '__main__':
    # 统计资源信息
    start = time.time()
    count = 0

    # 读取task_start_time中的提交时间
    reader = pd.read_csv(file_input_path, header=None, usecols=[0, 1])
    # 将读取到的提交时间转化为列表进行处理
    result_list = reader.values.tolist()
    print(result_list)
    print('提交的task数量为 %s' % len(result_list))
    duration_list = []
    for k in result_list:
        duration_list.append(k[1] - k[0])
    print('存在 %s 个task' % len(duration_list))
    save1 = pd.DataFrame({'task_duration2017': duration_list})
    save1.to_csv(file_output_path, index=False, header=False, mode='a')

    end = time.time()
    print('Running time: %s Seconds' % (end - start))
















