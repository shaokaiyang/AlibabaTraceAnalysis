# -*- coding: UTF-8 -*-
# 分析2017年trace数据
# 输入为：batch_task文件
# 输出为：每个job包含的task数量

import pandas as pd
import numpy as np
import time
import _thread


# 输入路径
file_input_path = 'D:\\AlibabaTraceAnalysis\\alibaba_clustertrace_2017\\batch_task.csv'
# 输出路径前缀
file_output_path = 'D:\\AlibabaTraceAnalysis\\AnalysisResult\\number_of_task_per_job2017.csv'


if __name__ == '__main__':
    # 统计资源信息
    start = time.time()

    # 读取task_start_time中的提交时间
    reader = pd.read_csv(file_input_path, header=None, usecols=[2])
    # 将读取到的提交时间转化为列表进行处理
    result_list = reader.sort_values(by=[2]).values.tolist()
    print(result_list)
    print('提交的task数量为 %s' % len(result_list))
    number_list = []
    last_time = result_list[0][0]
    count = 0
    for k in result_list:
        j = k[0]
        if j == last_time:
            count += 1
        else:
            number_list.append(count)
            last_time = j
            count = 1
    number_list.append(count)
    print('存在 %s 个job' % len(number_list))
    save1 = pd.DataFrame({'number_of_task_per_job2017': number_list})
    save1.to_csv(file_output_path, index=False, header=False, mode='a')

    end = time.time()
    print('Running time: %s Seconds' % (end - start))
















