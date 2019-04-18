# -*- coding: UTF-8 -*-

# 1 machine usage

import time
import pandas as pd
import numpy as np

# file_input_path = 'E:\\AlibabaTraceAnalysis\\test.csv'
file_input_path = 'E:\\AlibabaTraceAnalysis\\alibaba_clustertrace_2018\\batch_task.csv'

if __name__ == '__main__':
    start = time.time()
    # 统计资源信息
    reader = pd.read_csv(file_input_path, header=None, usecols=[0], chunksize=20000000)
    count_line = 0
    # instance_set = set()
    count = 0
    for chunk in reader:
        count_line += chunk.shape[0]
        count += 1
        print("第 %s 轮完成" % count)
        # if count == 2:
        #     break
    print(count_line)
    print(count)
    end = time.time()
    print('Running time: %s Seconds' % (end - start))


