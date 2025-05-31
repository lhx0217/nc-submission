import copy
import time

import numpy as np
from ipywidgets import interact


def min_distance(A, B):
    if not len(A) or not len(B):
        return [0] * len(A)
    A = np.array(A)
    B = np.array(B)
    A_squared = np.sum(A ** 2, axis=1, keepdims=True)
    B_squared = np.sum(B ** 2, axis=1, keepdims=True)
    AB_dot = np.dot(A, B.T)
    distances = np.sqrt(A_squared - 2 * AB_dot + B_squared.T)
    min_distances = np.min(distances, axis=1)
    return min_distances.tolist()


def covergence_test(airsim, m_l):
    pos = copy.deepcopy(airsim.pos)
    target = airsim.graph
    # 初始化目标点是否被覆盖的数组
    covered = np.zeros(len(target), dtype=bool)
    min_dis = min_distance(target, pos)
    # 从距离矩阵中选择最近的目标点，并分配无人机
    for i in range(len(target)):
        if min_dis[i] <= m_l + 0.2:
            covered[i] = True
    # 计算覆盖率
    coverage_rate = np.sum(covered) / len(target)
    return coverage_rate


def entering_test(airsim):
    pos = copy.deepcopy(airsim.pos)
    target = airsim.graph
    min_dis = min_distance(pos, target)

    # 初始化目标点是否被覆盖的数组
    entered = np.zeros(len(pos), dtype=bool)

    # 从距离矩阵中选择最近的目标点，并分配无人机
    for i in range(len(pos)):
        if min_dis[i] <= 1.1:
            entered[i] = True

    # 计算覆盖率
    entering_rate = np.sum(entered) / len(pos)

    # 获取没有进入的索引
    not_entered_indices = np.where(entered == False)[0]

    return entering_rate, not_entered_indices


def all_test(airsim, init_state, uav_list):
    km = init_state['kmeans']
    move = airsim.total_move
    avg_des = np.std(airsim.avg_des) / np.mean(airsim.avg_des)
    std_contain = np.std(airsim.container) / np.mean(airsim.container)
    std_dist = []
    command_percentage = [0, 0, 0, 0, 0, 0, 0, 0]
    interact, vel = [], []
    min_dist = []
    for uav in uav_list:
        std_dist += uav.record_nei_std
        command_percentage[0] += uav.enter / airsim.num
        command_percentage[1] += uav.explore / airsim.num
        command_percentage[2] += uav.contain_exp / airsim.num
        command_percentage[3] += uav.length_finetuned / airsim.num
        interact.append(uav.interact)
        vel.append(np.linalg.norm(uav.vel))
        min_dist.append(uav.min_dis_n)
        command_percentage[6] += uav.cmd_set / airsim.num

    command_percentage[4] = np.max(interact)
    command_percentage[5] = np.max(vel)
    command_percentage[7] = np.mean(np.linalg.norm(airsim.vel, axis=1))
    # std_dist2 = np.std(std_dist) / np.mean(std_dist)
    std_dist2 = np.mean(airsim.avg_dist)  # 表示平均距离的扩大
    e, non_enter = entering_test(airsim)
    c = covergence_test(airsim, np.mean(std_dist))
    u = np.std(min_dist) / np.mean(min_dist)
    return [e, c, u, move, avg_des, np.mean(vel), std_dist2, std_contain, np.min(min_dist), uav_list[0].round], command_percentage, non_enter
