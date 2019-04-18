# -*- coding: UTF-8 -*-

# 1 x:containerID y: cpuRequest,cpuLimit
# 2 x:containerID y: memRequest
# 3 x:appID y:include containers
# 4 x:machineID y:include containers,apps
# 5 container number, app number


import pandas as pd
import numpy as np


# file_input_path = 'E:\\AlibabaTraceAnalysis\\test.csv'
file_input_path = 'E:\\AlibabaTraceAnalysis\\alibaba_clustertrace_2018\\container_meta.csv'
file_output_path1 = 'E:\\AlibabaTraceAnalysis\\AnalysisResult\\container_cpuReq_cpuLim.csv'
file_output_path2 = 'E:\\AlibabaTraceAnalysis\\AnalysisResult\\container_memReq.csv'
file_output_path3 = 'E:\\AlibabaTraceAnalysis\\AnalysisResult\\onlineapp_include_containers.csv'
file_output_path4 = 'E:\\AlibabaTraceAnalysis\\AnalysisResult\\machine_include_apps_containers.csv'


# 统计container资源使用情况,参数为DF和文件行数
def statistic_container_meta_resource(data_frame, n):
    # 记录containerID所包含的信息CPU，mem
    container_resource_request_map = dict()
    # 记录containerID
    container_list = list()
    # 记录CPU申请
    container_cpuReq_list = list()
    # 记录CPU限制
    container_cpuLim_list = list()
    # 记录内存申请
    container_mem_list = list()
    for i in range(0, n):
        # 读取一行数据
        row_info = data_frame.loc[i]
        # container 申请的资源信息仅保留一次即可
        if row_info[0] not in container_resource_request_map:
            container_resource_request_map[row_info[0]] = [row_info[5], row_info[6], row_info[7]]
            container_list.append(row_info[0])
            container_cpuReq_list.append(row_info[5])
            container_cpuLim_list.append(row_info[6])
            container_mem_list.append(row_info[7])
    return [len(container_resource_request_map), container_list, container_cpuReq_list,
            container_cpuLim_list, container_mem_list]


# 统计app包含的container数量信息
def statistic_container_meta_app(data_frame, n):
    # 记录appID包含的container数量
    app_include_containers_map = dict()
    # 记录app
    app_list = list()
    # 记录app包含的container数量
    app_include_containers_list = list()
    for i in range(0, n):
        # 读取一行数据
        row_info = data_frame.loc[i]
        # 统计app包含的container数量
        if row_info[3] not in app_include_containers_map:
            app_include_containers_map[row_info[3]] = {row_info[0]}
        else:
            app_include_containers_map[row_info[3]].add(row_info[0])
    # 将统计的app中包含的container数量信息变为列表
    for tmp in app_include_containers_map:
        app_list.append(tmp)
        app_include_containers_list.append(len(app_include_containers_map[tmp]))
    return [len(app_include_containers_map), app_list, app_include_containers_list]


# 统计machine包含的app和container信息
def statistic_container_meta_machine(data_frame, n):
    # 记录machineID包含的container数量和app数量
    machine_include_apps_containers_map = dict()
    # 记录machine
    machine_list = list()
    # 记录machine包含的container数量
    machine_include_containers_list = list()
    # 记录machine包含的app数量
    machine_include_apps_list = list()
    for i in range(0, n):
        # 读取一行数据
        row_info = data_frame.loc[i]
        # 统计机器上包含的container和app数量
        if row_info[1] not in machine_include_apps_containers_map:
            machine_include_apps_containers_map[row_info[1]] = [{row_info[0]}, {row_info[3]}]
        else:
            machine_include_apps_containers_map[row_info[1]][0].add(row_info[0])
            machine_include_apps_containers_map[row_info[1]][1].add(row_info[3])
    # 将统计的machine中的app和container信息变为列表
    for tmp in machine_include_apps_containers_map:
        machine_list.append(tmp)
        machine_include_containers_list.append(len(machine_include_apps_containers_map[tmp][0]))
        machine_include_apps_list.append(len(machine_include_apps_containers_map[tmp][1]))
    # 返回container数目，app数目，containerID列表，资源申请列表，app包含的container数量，machine包含的app和container数量
    return [machine_list, machine_include_containers_list, machine_include_apps_list]


if __name__ == '__main__':
    # 统计资源信息
    df = pd.read_csv(file_input_path, header=None, usecols=[0, 5, 6, 7])
    # 记录文件行数
    max_length = df.shape[0]
    print(df.head())
    result = statistic_container_meta_resource(df, max_length)
    print(result[0])
    save1 = pd.DataFrame({'containerID': result[1], 'container_cpuReq': result[2], 'container_cpuLim': result[3]})
    save1.to_csv(file_output_path1, index=False)
    save2 = pd.DataFrame({'containerID': result[1], 'container_memReq': result[4]})
    save2.to_csv(file_output_path2, index=False)

    # 统计app包含container信息
    df = pd.read_csv(file_input_path, header=None, usecols=[0, 3])
    # 记录文件行数
    max_length = df.shape[0]
    result = statistic_container_meta_app(df, max_length)
    print(result[0])
    save3 = pd.DataFrame({'appID': result[1], 'app_include_containers': result[2]})
    save3.to_csv(file_output_path3, index=False)

    # 统计machine包含的app和container信息
    df = pd.read_csv(file_input_path, header=None, usecols=[0, 1, 3])
    # 记录文件行数
    max_length = df.shape[0]
    result = statistic_container_meta_machine(df, max_length)
    save4 = pd.DataFrame({'machineID': result[0], 'machine_include_containers': result[1],'machine_include_apps': result[2]})
    save4.to_csv(file_output_path4, index=False)
