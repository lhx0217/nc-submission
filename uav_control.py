import copy
import numpy as np
from collections import Counter
from scipy.spatial import KDTree
import airsim


class UAV():
    def __init__(self, num, sim, **kwargs):

        self.num = num  # 本机编号
        self.client = airsim.MultirotorClient()
        self.init_graph = np.array(kwargs['graph'], dtype=float)  # 初始以原点为中心的图
        self.sim = sim  # 控制的类
        self.pos = self.get_pos()  # 当前位置
        self.pos_color = 0  # 当前颜色, 0为白色, 1为黑色
        self.kmeans = kwargs['kmeans']  # 是否启用kmeans聚类辅助

        self.neigh_set = []  # 邻居位置
        self.neigh_vel = []  # 邻居速度
        self.neigh_contain = []  # 邻居的密度信息
        self.neigh_avg = []  # 邻居的平均距离
        self.neigh_avg_dist = []  # 邻居的真实平均距离
        self.nei_len = []  # 邻居的邻居数量
        self.neigh_indexing = []  # 临近邻居的索引
        self.nei_color = []

        self.graph_angle = np.array(kwargs['des_angle'])  # 本机存储的图形角度
        self.graph_vel = kwargs['graph_vel']  # 本机存储的图形速度
        self.graph_angle_vel = kwargs['angle_vel']  # 本机存储的图形角速度
        self.contain = 0  # 属于本机的方格数量

        self.graph_center = np.array(kwargs['des_pos'])  # 图形偏移
        self.graph = KDTree(copy.deepcopy(self.init_graph) + self.graph_center)  # 真实图
        self.graph_color = np.array(kwargs['graph_color'], dtype=float)  # 图上各点颜色
        # 函数间相互调用的变量
        self.indices_within_distance = None
        self.points_within_distance = None
        self.min_index = None

        self.vel = np.array([0, 0, 0], dtype=float)
        self.t = kwargs['t']
        self.t1, self.t2 = 0, 0
        # 这个问题怎么解决
        self.r_sense = kwargs['r_sense']
        self.r_avoid = kwargs['r_avoid']
        self.o_avoid = kwargs['r_avoid']

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

        self.nei_index_0 = []

        # self.state = None

    def add_noise(self, vec, sigma=0.05, delta=0.1):
        return vec + np.random.normal(0, sigma, size=vec.shape) + np.random.uniform(-delta, delta, size=vec.shape)


    def find_closer_points_large_data(self, target_points, neighbor_points, point_A):
        """
        找出在目标点集中距离点 A 更近于任何邻居点集中的点。

        :param target_points: 目标点的 numpy 数组（shape: [n, d]）
        :param neighbor_points: 邻居点集的 numpy 数组（shape: [m, d]）
        :param point_A: 参考点 A 的坐标（shape: [d]）
        :return: 符合条件的目标点
        """
        # Convert to numpy arrays if needed
        if not np.size(neighbor_points) or not np.size(target_points):
            return [], -1, None, None
        target_points_t = KDTree(neighbor_points)

        # Compute distances from point_A to all target points
        distances_from_A = np.linalg.norm(target_points - point_A, axis=1)

        # Determine the minimum distance to any neighbor point
        min_distances_from_neighbors, min_index = target_points_t.query(target_points)  # 行向量代表到最近的邻居的距离

        # Find points where distance to point_A is less than the minimum distance to any neighbor point
        closer_points_indices = np.where(distances_from_A <= min_distances_from_neighbors)[0]
        outer_points = np.where((-1.5 <= distances_from_A - min_distances_from_neighbors) &
                                (distances_from_A - min_distances_from_neighbors <= 1.5))[0]
        if np.size(closer_points_indices) == 0:
            return [], -1, None, None
        closer_nei_indices = list(set(min_index[closer_points_indices]))
        outer_indices = min_index[outer_points]
        counts = dict(Counter(outer_indices))
        dis = np.linalg.norm(neighbor_points[closer_nei_indices] - point_A, axis=1)
        sorted_pairs = sorted(zip(dis, closer_nei_indices))
        # 分别提取排序后的 dis 和 closer_nei_indices 列表
        dis, closer_nei_indices = zip(*sorted_pairs)
        kmeans = target_points[closer_points_indices]  # 覆盖范围内的所属点
        # 所属点, 所属点索引, 邻居索引, 邻居边长
        return kmeans, closer_points_indices, list(closer_nei_indices), \
            [counts[key] if key in counts.keys() else 0 for key in closer_nei_indices]

    def unit_vector(self, start, end):
        dx, dy, dz = end[0] - start[0], end[1] - start[1], end[2] - start[2]
        mag = (dx ** 2 + dy ** 2 + dz ** 2) ** 0.5
        return np.array([dx / mag, dy / mag, dz / mag], dtype=float) if mag else np.array([0.0, 0.0, 0.0])

    def find_closest_points(self, points, target, n):
        """
        找出距离目标点最近的n个点。

        参数:
        points -- 点的集合，格式为numpy数组，每行是一个点的坐标。
        target -- 目标点，格式为numpy数组。
        n -- 需要找出的最近点的数量。

        返回:
        最近点的索引和对应的距离。
        """
        distances = np.array([np.linalg.norm(p - target) for p in points])
        closest_n_indices = np.argsort(distances)[:n]
        return closest_n_indices

    def weight_function(self, x, r):
        y = (np.cos(np.pi * x / r) + 1) / 2
        # y = (cos_angle + 1) / 2
        y[x <= 0] = 1
        y[x > r] = 0
        return y

    def get_pos(self):
        # 清空记录
        self.contain_exp = 0
        self.length_finetuned = 0
        self.enter = 0
        self.interact = 0
        return self.sim.pos[self.num]

    def get_graph_pos_angle(self):
        return self.sim.get_graph_pos_angle_vel()

    def input_container(self, index, num):
        self.sim.input_container(index, num)

    # 需获取坐标和角度
    def GetNeighborSet(self):
        self.obstacles = None
        self.neigh_set, self.neigh_vel, self.neigh_contain, self.neigh_avg_contain, self.neigh_avg, self.neigh_avg_dist, self.nei_len, self.nei_color = self.sim.get_neighbor(
            self.num)
        self.neigh_set = self.add_noise(self.neigh_set)
        if len(self.neigh_set):
            self.min_dis_n = np.min(np.linalg.norm(self.pos - self.neigh_set, axis=1))
        else:
            self.min_dis_n = self.r_sense

    def get_pos_color(self):
        # 计算所有黑点到自身当前点的距离
        distance, self.min_index = self.graph.query(self.pos, k=1)
        self.pos_color = 1 if distance <= 1 else 0
        self.sim.pos_color[self.num] = self.pos_color
        # 找到距离小于给定阈值的点的索引
        ind = self.graph.query_ball_point(self.pos, self.r_sense)
        self.indices_within_distance = np.array(ind)
        if len(self.indices_within_distance):
            # 获取所有距离小于给定阈值的点
            self.points_within_distance = self.graph.data[self.indices_within_distance]
        else:
            self.points_within_distance = np.array([])

    def EnteringCmd(self):
        # 直接寻找传感范围内距离最近且颜色更深的点作为目标
        # 找到灰度值大于给定灰度值的点的索引
        if self.pos_color:
            if not np.size(self.neigh_vel):
                self.enter = 0
                return np.array([0, 0, 0], dtype=float)
            # print(f"跟随速度: ", - np.sum(self.vel - self.neigh_vel, axis=0) / len(self.neigh_vel))
            # other = - np.sum(self.vel - self.neigh_vel, axis=0) / len(self.neigh_vel)  # 有待确认
            # self.enter = np.linalg.norm(other)
            return np.zeros(3)  # 周围没有灰度点, 则跟随集群运动
        # 返回距离最近且灰度值大于给定灰度值的点的坐标和灰度值
        enter = (self.graph.data[self.min_index] - self.pos)

        self.enter = np.linalg.norm(enter)
        if self.enter > 2:
            enter = self.unit_vector([0,0,0], enter) * 2
        self.enter = np.linalg.norm(enter)
        return enter

    def ExplorationCmd(self, Kmeans=None, points_index=None, neigh_indexing0=None, edge_length=None):
        # 找到距离小于给定阈值的点的索引
        indices_within_distance = self.indices_within_distance
        points_within_distance = self.points_within_distance
        if not np.size(points_within_distance) or not len(self.neigh_set):
            return np.array([0, 0, 0], dtype=float)
        # 获取所有距离小于给定阈值的点
        avg_control = np.array([0.0, 0.0, 0.0])
        avg_explore = np.array([0.0, 0.0, 0.0])
        goal_set_neigh = np.array([0.0, 0.0, 0.0])

        if not neigh_indexing0 or len(neigh_indexing0) == 0:
            neigh_indexing0 = []
            edge_length = []
        self.nei_index_0 = neigh_indexing0
        print('邻居: ', neigh_indexing0)
        if len(neigh_indexing0):
            points_within_distance, indices_within_distance = Kmeans, indices_within_distance[points_index]
            self.points_within_distance, self.indices_within_distance = points_within_distance, indices_within_distance
            # closer_nei指的是nei_set中的元素，并非索引
            neigh_indexing = np.array(neigh_indexing0[:6] if len(neigh_indexing0) > 6 else neigh_indexing0)
            self.neigh_indexing = neigh_indexing
            avg_des0 = sum(np.linalg.norm(self.pos - self.neigh_set[neigh_indexing], axis=1)) / len(
                self.neigh_set[neigh_indexing])
            avg_des = (sum(self.neigh_avg) + avg_des0) / (len(self.neigh_avg_dist) + 1)  # neigh_avg / neigh_avg_dist
            self.sim.avg_dist[self.num] = avg_des0
            self.sim.nei_len[self.num] = len(neigh_indexing)
            if self.round == 0:
                self.sim.avg_des[self.num] = avg_des0
            else:
                self.sim.avg_des[self.num] = avg_des
            self.record_nei_std = []
            for nei in self.neigh_set[neigh_indexing]:
                self.record_nei_std.append(np.linalg.norm(self.pos - nei))
                avg_control += (np.linalg.norm(self.pos - nei) - avg_des) * self.unit_vector(self.pos, nei)
            # for nei, nei_avg, nei_dis, nei_len in zip(self.neigh_set[neigh_indexing], self.neigh_avg[neigh_indexing],
            #                                           self.neigh_avg_dist[neigh_indexing],
            #                                           self.nei_len[neigh_indexing]):
            #     if self.round > 0:
            #         avg_explore += (nei_dis - nei_avg) / nei_len * self.unit_vector(self.pos, nei) + (
            #                 avg_des0 - avg_des) / len(neigh_indexing) * self.unit_vector(self.pos, nei)
            for nei, nei_dis, color in zip(self.neigh_set, self.neigh_avg_dist, self.nei_color):
                if color:
                    avg_explore += (nei_dis - avg_des0) * self.unit_vector(self.pos, nei)

            self.contain = len(Kmeans)
            self.input_container(self.num, len(Kmeans))
            if len(edge_length):
                edge_length = edge_length[:6] if len(edge_length) > 6 else edge_length
                for nei, edge in zip(neigh_indexing, edge_length):
                    goal_set_neigh += edge * (self.neigh_contain[nei] - self.contain) * self.unit_vector(self.pos,
                                                                                                         self.neigh_set[
                                                                                                             nei])
                if len(neigh_indexing) and sum(edge_length):
                    goal_set_neigh /= len(neigh_indexing)
                    goal_set_neigh /= sum(edge_length)

                if self.round == 0:
                    self.sim.avg_contain[self.num] = self.contain
                    goal_set_neigh = np.array([0.0, 0.0, 0.0])
                else:
                    self.sim.avg_contain[self.num] = (np.sum(
                        self.neigh_avg_contain[neigh_indexing]) + self.contain) / (
                                                                len(neigh_indexing) + 1)
                    if np.linalg.norm(goal_set_neigh) > 1:
                        goal_set_neigh = self.unit_vector(np.array([0, 0, 0]), goal_set_neigh) * 1

        if not len(list(points_within_distance)):
            self.length_finetuned = np.linalg.norm((avg_explore + avg_control))
            self.contain_exp = np.linalg.norm(goal_set_neigh)
            return (avg_explore + avg_control) + goal_set_neigh
        vec_within_distance = self.pos - points_within_distance
        self.record_pos.append(copy.deepcopy(list(self.pos)))

        dis = np.linalg.norm(vec_within_distance, axis=1)
        # 计算每个黑点到每个邻居点的距离
        distances_neigh = np.linalg.norm(points_within_distance[:, None, :] - self.neigh_set, axis=2)
        # 判断是否在任一邻居的球内
        inside_any_sphere = np.any(distances_neigh < 0.2, axis=1)
        # 返回不在任一邻居躲避范围内的坐标
        points_without_occupy = points_within_distance[~inside_any_sphere]
        if not len(points_without_occupy):
            self.length_finetuned = np.linalg.norm((avg_explore + avg_control)*3)
            self.contain_exp = np.linalg.norm(goal_set_neigh)
            print('return 1')
            return (avg_explore + avg_control) * 3 + goal_set_neigh
        # 取符合条件的邻居计算权重
        weigh = self.weight_function(dis[~inside_any_sphere], self.r_sense)
        goal_set_expl = np.sum(points_without_occupy * weigh[:, None], axis=0) / sum(weigh)
        assert len(weigh) != 0, "division by zero 312"
        cmd_set = (goal_set_expl - self.pos)
        self.contain_exp = np.linalg.norm(goal_set_neigh * 2)
        self.length_finetuned = np.linalg.norm((3 * avg_control + avg_explore / 2))
        self.explore = np.linalg.norm(cmd_set * 2)
        return cmd_set * 2 + (avg_control * 3 + avg_explore / 2) + goal_set_neigh * 2

    def InteractionCmd_n(self, r_avoid):
        if not len(self.neigh_set):
            return np.array([0, 0, 0])
        distances = np.linalg.norm(self.neigh_set - self.pos, axis=1)
        indices_within_distance = np.where(distances < r_avoid)[0]
        if not len(indices_within_distance):
            return np.array([0, 0, 0])
        points_within_ravoid = self.neigh_set[indices_within_distance]
        weigh = r_avoid / distances[indices_within_distance] - 1
        vec = (self.pos - points_within_ravoid) / distances[indices_within_distance].reshape(-1, 1)
        cmd_set = np.sum(vec * weigh.reshape(-1, 1), axis=0) / sum(weigh)
        return cmd_set

    def op_uav_cmd(self, cmd_set, num):
        if self.round == 0:
            return
        vel = (self.sim.vel[num]) * 0.3 + cmd_set * 0.7
        pos = copy.deepcopy(self.sim.pos[num])
        pos += vel * self.t
        # state = self.client.getMultirotorState(vehicle_name='UAV' + str(num))l
        self.sim.total_move += np.linalg.norm(pos - self.sim.pos[num])
        self.sim.pos[num] = pos
        self.vel = vel
        self.sim.vel[num] = vel


    def run(self, round):
        print("===============", self.round, self.num, "===============")
        self.round = round
        self.pos = self.get_pos()
        self.pos = self.add_noise(self.pos)
        self.graph, self.graph_vel = self.get_graph_pos_angle()
        if round == 1 and self.num == 0:
            with open('./data/graph.json', 'w+') as f:
                f.write(str(self.graph.data.tolist()))
        self.graph_color = np.array([1] * len(self.graph.data))
        self.get_pos_color()
        self.GetNeighborSet()
        cmd_enter = self.EnteringCmd()
        Kmeans, points_index, neigh_indexing0, edge_length = self.find_closer_points_large_data(self.points_within_distance,
                                                                                                self.neigh_set,
                                                                                                self.pos)
        cmd_explore = self.ExplorationCmd(Kmeans, points_index, neigh_indexing0, edge_length)
        cmd_interact = self.InteractionCmd_n(self.r_avoid)
        cmd_set = cmd_enter * np.max([0, 1-np.linalg.norm(cmd_interact)]) + cmd_interact + cmd_explore * np.max([0, 1-np.linalg.norm(cmd_interact)])

        self.cmd_set = np.linalg.norm(cmd_set)
        v = 2
        if len(self.neigh_set):
            # av_points = self.avoid_points(cmd_set)
            if self.pos_color:
                if self.contain and len(self.neigh_contain):
                    m_contain = np.mean(
                        np.abs(self.neigh_contain[neigh_indexing0] - self.contain)) / self.contain * 3
                else:
                    m_contain = 2
                m_l = abs(self.sim.avg_des[self.num] - self.sim.avg_dist[self.num]) * 3
            else:
                m_contain, m_l = 2, 2
            v = min(max(m_contain, m_l), 2)
        if self.cmd_set > v:
            cmd_set = self.unit_vector(np.array([0, 0, 0]), cmd_set) * v
        self.interact = np.linalg.norm(cmd_interact)
        self.op_uav_cmd(cmd_set, self.num)
        self.cmd_set = np.linalg.norm(cmd_set)