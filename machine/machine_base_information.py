# -*- coding: UTF-8 -*-

import numpy as np
import pandas as pd

file_input_path = 'E:\\AlibabaTraceAnalysis\\alibaba_clustertrace_2018\\machine_meta.csv'
file_output_path1 = 'E:\\AlibabaTraceAnalysis\\AnalysisResult\\machine_frame_result.csv'
file_output_path2 = 'E:\\AlibabaTraceAnalysis\\AnalysisResult\\machine_room_result.csv'


class MachineMetaStructure:
    """
    记录机器的元数据信息
    @:param machine_id: 机器的ID信息
    @:param machine_timestamp: 机器的时间戳信息
    @:param machine_frame: 机器所在的机架编号
    @:param machine_room: 机器所在的机房编号
    """
    def __init__(self, machine_id, machine_timestamp, machine_frame, machine_room):
        self.machine_id = machine_id
        self.machine_timestamp = machine_timestamp
        self.machine_frame = machine_frame
        self.machine_room = machine_room


# 逐行读取数据并进行统计处理，统计机器数目，机架上的机器数目，机房内的机器数目
def statistic_machine_meta(data_frame):
    # 记录最大的机架序列
    max_machine_frame = data_frame.iloc[:, 2].max()
    # 记录最大的机房序列
    max_machine_room = data_frame.iloc[:, 3].max()
    # 记录文件行数
    max_length = data_frame.shape[0]
    # 记录机器信息的set集合
    machine_set = set()
    # 记录机架机器数量的map集合
    machine_frame_map = dict()
    # 记录机房数量的map集合
    machine_room_map = dict()
    # 记录机架机器数量的序列
    machine_frame_list = [0] * max_machine_frame
    # 记录机房机器数量序列
    machine_room_list = [0] * int(max_machine_room + 1)
    for i in range(max_length):
        # 读取一行数据
        row_info = data_frame.loc[i]
        # 进行无效数据过滤，不符合时间戳或者为NAN
        if row_info[1] not in range(1, 691201) \
                or row_info[2] not in range(1, max_machine_frame + 1) \
                or row_info[3] < 1.0 \
                or row_info[3] > max_machine_room \
                or np.isnan(row_info[3]):
            continue
        # 将一行数据存储到结构体中进行统计
        machine_info = MachineMetaStructure(row_info[0], row_info[1], row_info[2], row_info[3])
        # 符合时间戳要求的进行统计
        int(machine_info.machine_room)
        # 统计机器数目
        machine_set.add(machine_info.machine_id)
        # 统计机架上的机器数目
        if machine_info.machine_frame not in machine_frame_map:
            machine_frame_map[machine_info.machine_frame] = {machine_info.machine_id}
        else:
            machine_frame_map[machine_info.machine_frame].add(machine_info.machine_id)
        # 统计机房中机器数目
        if machine_info.machine_room not in machine_room_map:
            machine_room_map[machine_info.machine_room] = {machine_info.machine_id}
        else:
            machine_room_map[machine_info.machine_room].add(machine_info.machine_id)
    # 构建机架数目和机房数目的list
    for key in machine_frame_map.keys():
        machine_frame_list[key - 1] = len(machine_frame_map[key])
    for key in machine_room_map.keys():
        machine_room_list[int(key)] = len(machine_room_map[key])
    # 0：存储机器数量信息；1：存储不同机架上的机器数量；2：存储不同机房内的机器数量
    return [len(machine_set), machine_frame_list, machine_room_list]


if __name__ == '__main__':
    # 读入CSV文件
    df = pd.read_csv(file_input_path)
    # print(df.iloc[:, 2].min())
    # print(df.iloc[:, 2].max())
    # print(df.iloc[:, 3].min())
    # print(df.iloc[:, 3].max())
    # 获取数据
    result = statistic_machine_meta(df)
    print(result[0])
    # 写入文件
    save1 = pd.DataFrame({'machine_frame': result[1]})
    save1.to_csv(file_output_path1, index=False)
    save2 = pd.DataFrame({'machine_room': result[2]})
    save2.to_csv(file_output_path2, index=False)
