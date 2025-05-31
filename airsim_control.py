import copy
import math
import os
import time

import numpy as np
import airsim
from scipy.spatial import KDTree

'''避障还需要再考虑考虑，目前会来回震荡'''


class AirSim():
    def __init__(self, **kwargs):
        self.init_pos = np.array(kwargs['pos'], dtype=float)  # 初始偏移
        self.num = len(kwargs['pos'])  # 无人机的数量
        self.pos = np.array(kwargs['pos'], dtype=float)  # 所有无人机的位置
        self.vel = np.zeros((self.num, 3), dtype=float)  # 自身的速度
        self.r_sense = kwargs['r_sense']  # 传感距离
        self.init_graph = kwargs['graph']  # 图在原点处的样子
        self.graph = kwargs['graph']  # 图经过旋转平移和障碍占用后的样子
        self.kd_graph = None
        self.obstacles = kwargs['obstacles']  # 障碍物
        self.graph_center = np.array(kwargs['des_pos'], dtype=float)  # 存储图的中心位置
        self.graph_vel = np.array(kwargs['graph_vel'], dtype=float)  # 存储图的速度
        self.graph_angle = np.array(kwargs['des_angle'], dtype=float)  # 本机存储的图形角度
        self.angle_vel = kwargs['angle_vel']  # 存储图的角速度
        self.destination = np.array(kwargs['destination'], dtype=float)

        self.container = np.zeros(self.num, dtype=int)  # 包含的方格数量
        self.avg_contain = np.zeros(self.num, dtype=int)
        self.avg_des = np.ones(self.num, dtype=float)
        self.avg_dist = np.ones(self.num, dtype=float)
        self.nei_len = np.ones(self.num, dtype=float)
        self.pos_color = np.zeros(self.num, dtype=float)
        self.t1, self.t2 = 0, 0  # 时间间隔
        self.total_move = 0

        self.record_rate = [[0, 0, 1]]
        self.current_time = 0

    def get_graph_pos_angle_vel(self):
        toward = self.destination - self.graph_center
        if np.linalg.norm(toward) > 0:
            self.graph_vel = toward / np.linalg.norm(toward) * 0
        elif np.linalg.norm(toward) > 0:
            self.graph_vel = toward
        else:
            self.graph_vel = np.zeros(3)
        return self.kd_graph, self.graph_vel

    def GetDynFormation(self):
        angles = np.radians(self.graph_angle)
        # 定义旋转矩阵
        rotation_x = np.array([[1, 0, 0],
                               [0, np.cos(angles[0]), -np.sin(angles[0])],
                               [0, np.sin(angles[0]), np.cos(angles[0])]])

        rotation_y = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                               [0, 1, 0],
                               [-np.sin(angles[1]), 0, np.cos(angles[1])]])

        rotation_z = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                               [np.sin(angles[2]), np.cos(angles[2]), 0],
                               [0, 0, 1]])

        # 对所有点进行旋转操作
        rotated_points = np.dot(rotation_x, np.dot(rotation_y, np.dot(rotation_z, self.init_graph.T))).T
        self.graph = rotated_points + self.graph_center
        self.kd_graph = KDTree(self.graph)
        print('graph_Center: ', self.graph_center, self.graph_vel)
        print('pos_Center: ', np.mean(self.pos, axis=0))


    def get_neighbor(self, num):
        target_point = self.pos[num]
        pos = np.array(self.pos)  # 将所有点的位置转换为NumPy数组
        distance_to_target = np.linalg.norm(pos - target_point, axis=1)  # 计算所有点到目标点的距离
        mask = (distance_to_target < self.r_sense) & (np.arange(self.num) != num)  # 创建一个掩码，用于筛选出距离小于阈值且不是目标点的点

        # 使用掩码一次性提取所有需要的数据
        within_distance_points = self.pos[mask]
        within_distance_vel = self.vel[mask]
        container = self.container[mask]
        avg_contain = self.avg_contain[mask]
        avg_des = self.avg_des[mask]
        avg_dist = self.avg_dist[mask]
        nei_len = self.nei_len[mask]
        pos_color = self.pos_color[mask]
        return np.array(within_distance_points), np.array(within_distance_vel), np.array(container), np.array(
            avg_contain), np.array(avg_des), np.array(avg_dist), np.array(nei_len), np.array(pos_color)

    def input_container(self, index, num):
        self.container[index] = num

    def op_graph_cmd_o(self):
        if self.t1!=0:
            delta_t = (self.t2-self.t1) / 1000000000
            self.graph_angle += self.graph_angle * delta_t