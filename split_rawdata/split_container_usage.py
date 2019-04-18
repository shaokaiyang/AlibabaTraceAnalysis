# -*- coding: UTF-8 -*-
# 因原始数据较大且首尾存在波动值，对 machine_usage 文件进行切分，选取第4,5，6天
# 本研究关注机器资源使用情况，故对于文件中的信息仅保留machineID,timestamp,cpuutil,memutil

import pandas as pd
import numpy as np
import time
import _thread

day = 86400

# file_input_path = 'E:\\AlibabaTraceAnalysis\\test.csv'
# 输入路径
file_input_path = 'D:\\AlibabaTraceAnalysis\\alibaba_clustertrace_2018\\container_usage.csv'
# 输出路径前缀
file_output_prefix = 'D:\\AlibabaTraceAnalysis\\SplitResult\\container_usage_'
# 需要保留的列
save_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# 时间戳所在列
timestamp_index = 2


# 数据框架、时间戳所在列索引，天索引，输出路径
def group_write(data_frame, data_index, index, file_out_path):
    save = data_frame[(data_frame[data_index] >= day*index) & (data_frame[data_index] <= day*(index+1)-1)]
    save.to_csv(file_out_path, index=False, header=False, mode='a')


if __name__ == '__main__':
    # 统计资源信息
    start = time.time()
    reader = pd.read_csv(file_input_path, header=None, usecols=save_list, chunksize=20000000)
    count = 0
    for chunk in reader:
        for i in range(0, 3):
            file_path = file_output_prefix + str(i) + '.csv'
            # _thread.start_new_thread(group_write, (chunk, 5, i+3, file_path))
            group_write(chunk, timestamp_index, i+3, file_path)
        count += chunk.shape[0]
        print("前 %s 行完成" % count)
        # if count == 10000:
        #     break
    print(count)
    end = time.time()
    print('Running time: %s Seconds' % (end - start))




