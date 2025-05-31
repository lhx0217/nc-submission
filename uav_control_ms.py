import copy
import numpy as np
import airsim
from scipy.spatial import KDTree


class UAV():
    def __init__(self, num, informed, sim, **kwargs):
        self.client = airsim.MultirotorClient()
        self.init_graph = np.array(kwargs['graph'], dtype=float)  # 初始以原点为中心的图
        self.informed = np.array(informed, dtype=int)  # 是否是领导者
        self.des_pos = np.array(kwargs['des_pos'])  # 目标中心
        self.des_angle = np.array(kwargs['des_angle'])  # 目标角度
        self.num = num  # 本机编号
        self.sim = sim  # 控制的类
        self.state = None
        self.pos = None
        self.pos_color = 0  # 当前颜色, 0为白色, 1为黑色

        self.neigh_set = []  # 邻居位置
        self.neigh_vel = []  # 邻居速度
        self.neigh_graph_pos = []  # 邻居存储图形位置
        self.neigh_graph_vel = []  # 邻居存储图形速度
        self.neigh_angle = []  # 邻居存储图形角度
        self.neigh_angle_vel = []  # 邻居存储图形角速度

        self.graph_angle = np.array([0, 0, 0], dtype=float)  # 本机存储的图形角度
        self.graph_vel = np.array([0, 0, 0], dtype=float)  # 本机存储的图形速度
        self.graph_angle_vel = np.array([0, 0, 0], dtype=float)  # 本机存储的图形角速度

        self.graph = np.array(copy.deepcopy(self.init_graph), dtype=float)  # 真实图
        self.graph_color = np.array(kwargs['graph_color'], dtype=float)  # 图上各点颜色
        self.graph_center = np.array(copy.deepcopy(self.pos), dtype=float)  # 图形偏移
        self.GetDynFormation()
        self.kd_graph = KDTree(self.graph)

        self.vel = np.array([0, 0, 0], dtype=float)
        self.t = kwargs['t']
        # 这个问题怎么解决
        self.r_sense = kwargs['r_sense']
        self.r_avoid = kwargs['r_avoid']

        self.cmd_set = 0
        self.contain_exp = 0
        self.length_finetuned = 0
        self.enter = 0
        self.interact = 0
        self.total_control = 0
        self.min_dis_n = 3
        self.explore = 0

        self.record_pos = []
        self.record_target = []
        self.record_nei_std = []
        self.round = 0

    def weight_function(self, x, r, s):
        # x是真实距离, r是单位1
        y = (1 + np.cos(np.pi * (x - 2 * s) / (r - 2 * s))) / 2
        y[x < 2 * s] = 1
        y[x > r] = 0
        return y

    def get_pos(self):
        pos = self.sim.pos[self.num]
        return pos

    def get_pos_color(self):
        distance, _ = self.kd_graph.query(self.pos, k=1)
        if distance < 1:
            # 全都不在, 为白色灰度为0
            self.pos_color = 1
        else:
            self.pos_color = 0

    def get_graph_pos_angle(self):
        return self.sim.get_graph_pos_angle_vel(self.num)

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
        self.graph = rotated_points + self.des_pos
        if self.num == 1:
            self.sim.graph = self.graph

    # 需获取坐标和角度
    def GetNeighborSet(self):
        self.neigh_set, self.neigh_vel, self.neigh_graph_pos, self.neigh_graph_vel, \
            self.neigh_angle, self.neigh_angle_vel = self.sim.get_neighbor(self.num)
        if len(self.neigh_set):
            self.min_dis_n = np.min(np.linalg.norm(self.pos - self.neigh_set, axis=1))
        else:
            self.min_dis_n = self.r_sense

    def NegotPositState(self):
        if not len(self.neigh_set):
            print('没有邻居')
            self.graph_vel = np.subtract(self.des_pos, self.pos) * self.informed
            return
        # 此处传感探测不到邻居会报错, 此处直接等于邻居速度均值, 而非加法
        avg_dis = np.sum([np.subtract(pos, self.graph_center) for pos in self.neigh_graph_pos], axis=0) / len(self.neigh_graph_pos)
        # 此处可能需要更改
        avg_vel = np.sum(self.neigh_graph_vel, axis=0) / len(self.neigh_graph_vel)
        cmd_set = avg_dis * (1-self.informed) + np.subtract(self.des_pos, self.graph_center) * self.informed + avg_vel * (1-self.informed)
        self.sim.op_graph_cmd_v(cmd_set, self.num)


    def NegotOrientState(self):
        if not len(self.neigh_angle):
            self.graph_angle_vel = 0
            return
        avg_angle = np.sum([np.subtract(angle, self.graph_angle) for angle in self.neigh_angle], axis=0) / len(self.neigh_angle)
        # 此处可能需要更改
        avg_angle_vel = np.sum(self.neigh_angle_vel, axis=0) / len(self.neigh_angle)
        cmd_set = avg_angle * (1-self.informed) + np.subtract(self.des_angle, self.graph_angle) * self.informed + avg_angle_vel
        self.sim.op_graph_cmd_o(cmd_set, self.num)

    def EnteringCmd(self):
        self.get_pos_color()
        # 直接寻找传感范围内距离最近且颜色更深的点作为目标
        # 计算点与所有灰度点的距离
        # print(f'位置颜色{self.pos_color}')
        # print(f'图形中心{self.graph_center}')
        # print(f'当前位置{self.pos}')
        if np.size(self.neigh_vel):
            print(f"跟随速度: ", - np.sum(self.vel - self.neigh_vel, axis=0) / len(self.neigh_vel))
            other = - np.sum(self.vel - self.neigh_vel, axis=0) / len(self.neigh_vel)
            dis = np.linalg.norm(self.pos - self.neigh_set, axis=1)
            min_three_indices = np.argsort(dis)[:2]  # 排序后取前3
            min_three_values = dis[min_three_indices]
            self.record_nei_std = min_three_values.tolist()
        else:
            other = np.zeros(3)
            self.record_nei_std = [1]

        dis, nearest_index = self.kd_graph.query(self.pos)

        # 返回距离最近且灰度值大于给定灰度值的点的坐标和灰度值
        nearest_point = self.graph[nearest_index]
        enter = nearest_point - self.pos
        if np.linalg.norm(enter) > 2:
            enter = self.unit_vector([0,0,0], enter) * 2
        return enter + other

    def ExplorationCmd(self):
        # 获取有效灰度点(纯黑的), 按照权重计算目标点

        # # 计算所有黑点到自身当前点的距离
        #
        # 找到距离小于给定阈值的点的索引
        indices_within_distance = self.kd_graph.query_ball_point(self.pos, self.r_sense)
        vaild_point = self.graph[indices_within_distance]
        distances = np.linalg.norm(vaild_point - self.pos, axis=1)
        # 获取所有距离小于给定阈值的点
        points_within_distance = vaild_point
        # print(f'范围内的所有点: {points_within_distance}')
        if not len(points_within_distance):
            print('no points')
            return np.array([0, 0, 0])
        weigh = self.weight_function(distances, self.r_sense, 0)
        # figplot.plot_3d_points(points_within_distance, "points_within_distance")
        # 将每个元素都乘以给定值
        # 将结果按列相加得到最终结果
        goal_set_fill = np.sum(points_within_distance * weigh[:, None], axis=0) / sum(weigh)

        if not len(self.neigh_set):
            return goal_set_fill - self.pos
        # 计算每个黑点到每个邻居点的距离
        distances_neigh = np.linalg.norm(points_within_distance[:, None, :] - self.neigh_set, axis=2)
        # 判断是否在任一邻居的球内
        inside_any_sphere = np.any(distances_neigh <= self.r_avoid, axis=1)
        # 返回不在任一邻居躲避范围内的坐标
        points_without_occupy = points_within_distance[~inside_any_sphere]
        if not len(points_without_occupy):
            print('no pints')
            return goal_set_fill - self.pos
        # 取符合条件的邻居计算权重
        weigh2 = self.weight_function(distances[~inside_any_sphere], self.r_sense, 0)
        goal_set_expl = np.sum(points_without_occupy * weigh2[:, None], axis=0) / sum(weigh2)
        cmd_set = (goal_set_fill - self.pos) + 5 * (goal_set_expl - self.pos)
        return cmd_set

    def InteractionCmd(self):
        if not len(self.neigh_set):
            return np.array([0, 0, 0])
        distances = np.linalg.norm(self.neigh_set - self.pos, axis=1)
        # print(f'邻居速度: {self.neigh_vel}')
        indices_within_distance = np.where(distances < self.r_avoid)[0]
        if not len(indices_within_distance):
            return np.array([0, 0, 0])
        points_within_ravoid = self.neigh_set[indices_within_distance]
        weigh = self.r_avoid / distances[indices_within_distance] - 1
        unit_vec = (self.pos - points_within_ravoid) / distances[indices_within_distance][:, np.newaxis]
        cmd_set = np.sum(weigh[:, np.newaxis] * unit_vec, axis=0)
        # cmd_set = np.array([0, 0, 0])
        if min(self.record_nei_std) < 0.2:
            print('avoid: ', cmd_set)
        return cmd_set

    def op_uav_cmd(self, cmd_set, num):
        if self.round == 0:
            return
        vel = (self.sim.vel[num]) * 0.3 + (cmd_set) * 0.7
        pos = copy.deepcopy(self.sim.pos[num])
        pos += vel * self.t
        # state = self.client.getMultirotorState(vehicle_name='UAV' + str(num))l
        self.sim.total_move += np.linalg.norm(pos - self.sim.pos[num])
        self.sim.pos[num] = pos
        self.vel = vel
        self.sim.vel[num] = vel

    def unit_vector(self, start, end):
        dx, dy, dz = end[0] - start[0], end[1] - start[1], end[2] - start[2]
        mag = (dx ** 2 + dy ** 2 + dz ** 2) ** 0.5
        return np.array([dx / mag, dy / mag, dz / mag], dtype=float) if mag else np.array([0.0, 0.0, 0.0])


    def run(self, round):
        print(f'=========={self.num} round {round}==========')

        self.round = round
        self.pos = self.get_pos()
        self.graph_center, self.graph_angle, self.graph_vel, self.graph_angle_vel\
            = self.get_graph_pos_angle()
        self.GetNeighborSet()
        self.NegotPositState()
        self.NegotOrientState()
        self.graph_center, self.graph_angle, self.graph_vel, self.graph_angle_vel\
            = self.get_graph_pos_angle()
        self.GetDynFormation()
        cmd_enter = self.EnteringCmd()
        cmd_explore = self.ExplorationCmd()
        cmd_interact = self.InteractionCmd()
        cmd_set = cmd_enter + cmd_explore + cmd_interact
        if np.linalg.norm(cmd_set) > 2:
            cmd_set = self.unit_vector(np.array([0, 0, 0]), cmd_set) * 2
        self.op_uav_cmd(cmd_set, self.num)