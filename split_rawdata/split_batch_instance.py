# -*- coding: UTF-8 -*-
# 因原始数据较大且首尾存在波动值，对 machine_usage 文件进行切分，选取第4,5，6天
# 本研究关注机器资源使用情况，故对于文件中的信息仅保留machineID,timestamp,cpuutil,memutil

import pandas as pd
import numpy as np
import time
import _thread

day = 86400

# file_input_path = 'E:\\AlibabaTraceAnalysis\\test.csv'
file_input_path = 'D:\\AlibabaTraceAnalysis\\alibaba_clustertrace_2018\\batch_instance.csv'


def group_write(data_frame, data_index, index, file_out_path):
    save = data_frame[(data_frame[data_index] >= day*index) & (data_frame[data_index] <= day*(index+1)-1)]
    save.to_csv(file_out_path, index=False, header=False, mode='a')


if __name__ == '__main__':
    # 统计资源信息
    start = time.time()
    reader = pd.read_csv(file_input_path, header=None, usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13], chunksize=20000000)
    count = 0
    for chunk in reader:
        for i in range(0, 3):
            file_path = 'D:\\AlibabaTraceAnalysis\\SplitResult\\batch_instance_%s.csv' % i
            # _thread.start_new_thread(group_write, (chunk, 5, i+3, file_path))
            group_write(chunk, 5, i+3, file_path)
        count += chunk.shape[0]
        print("前 %s 行完成" % count)
        # if count == 20000000:
        #     break
    print(count)
    end = time.time()
    print('Running time: %s Seconds' % (end - start))




