"""
test python environment
"""
import math
import threading
import time

import numpy as np
import airsim

class AirSim():
    def __init__(self, **kwargs):
        self.init_pos = np.array(kwargs['pos'], dtype=float)  # 初始偏移
        self.num = len(kwargs['pos'])  # 无人机的数量
        self.pos = np.array(kwargs['pos'], dtype=float)  # 所有无人机的位置
        self.vel = np.zeros((self.num, 3), dtype=float)  # 自身的速度
        self.r_sense = kwargs['r_sense']  # 传感距离
        self.graph = kwargs['graph']  # 图经过旋转平移和障碍占用后的样子
        self.graph_pos = np.array(self.pos, dtype=float)  # 存储图的角度
        self.graph_vel = np.zeros((self.num, 3), dtype=float)  # 存储图的角速度
        self.angle = np.zeros((self.num, 3), dtype=float)  # 存储图的角度
        self.angle_vel = np.zeros((self.num, 3), dtype=float)  # 存储图的角速度
        self.t = kwargs['t']  # 时间间隔
        self.avg_dist = np.ones(self.num, dtype=float)
        self.container = np.zeros(self.num, dtype=int)  # 包含的方格数量
        self.total_move = 0
        self.avg_des = np.ones(self.num, dtype=float)
        # p = threading.Thread(target=self.record)
        # p.start()

    def get_graph_pos_angle_vel(self, num):
        return self.graph_pos[num], self.angle[num], self.graph_vel[num], self.angle_vel[num]

    def get_neighbor(self, num):
        target_point = self.pos[num]
        within_distance_points = []
        within_distance_vel = []
        within_distance_graph_angle_vel = []
        within_distance_graph_angle = []
        within_distance_graph_pos = []
        within_distance_graph_vel = []
        for index in range(self.num):
            if index == num:
                continue
            # 计算点与目标点的距离
            point_x, point_y, point_z = self.pos[index]
            distance_to_target = math.sqrt((point_x - target_point[0]) ** 2 + (point_y - target_point[1]) ** 2
                                           + (point_z - target_point[2]) ** 2)
            # 如果距离小于指定距离，将点添加到结果列表中
            if distance_to_target < self.r_sense and index != num:
                within_distance_points.append(self.pos[index])
                within_distance_vel.append(self.vel[index])
                within_distance_graph_angle.append(self.angle[index])
                within_distance_graph_angle_vel.append(self.angle_vel[index])
                within_distance_graph_pos.append(self.graph_pos[index])
                within_distance_graph_vel.append(self.graph_vel[index])
        return np.array(within_distance_points), np.array(within_distance_vel), \
            np.array(within_distance_graph_pos), np.array(within_distance_graph_vel), \
            np.array(within_distance_graph_angle), np.array(within_distance_graph_angle_vel)



    def op_graph_cmd_v(self, cmd_set, num):
        self.graph_vel[num] = cmd_set
        self.graph_pos[num] += self.graph_vel[num] * 0.06

    def op_graph_cmd_o(self, cmd_set, num):
        self.angle_vel[num] = cmd_set
        self.angle[num] += self.angle_vel[num] * 0.06
