import copy

import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # 在导入 pyplot 前添加此配置
import matplotlib.pyplot as plt
import scipy

import settings


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

def covergence_test(pos, target):
    # 初始化目标点是否被覆盖的数组
    covered = np.zeros(len(target), dtype=bool)
    min_dis = min_distance(target, pos)
    # 从距离矩阵中选择最近的目标点，并分配无人机
    for i in range(len(target)):
        if min_dis[i] <= 3.7:
            covered[i] = True
    # 计算覆盖率
    coverage_rate = np.sum(covered) / len(target)
    return coverage_rate


class FormationGraph:
    """编队图结构建模（对应论文IV.A节）"""
    def __init__(self, positions: np.ndarray):
        # 输入：无人机3D位置矩阵 (N,3)
        self.positions = positions.astype(np.float64) / 50
        self.N = self.positions.shape[0]
        self.A = self._compute_adjacency()  # 邻接矩阵（公式1）
        self.D = self._compute_degree()  # 度矩阵（公式2）
        self.L = self.D - self.A
        # self.L_hat = self._compute_normalized_laplacian()  # 归一化拉普拉斯矩阵（公式4）
        pass

    def _compute_adjacency(self) -> np.ndarray:
        """公式1：A_ij = ||p_i - p_j||^2"""
        A = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(self.N):
                if i != j:
                    A[i, j] = np.sqrt(np.sum((self.positions[i] - self.positions[j]) ** 2))
        return A

    def _compute_degree(self) -> np.ndarray:
        """公式2：D_ii = Σ_j A_ij"""
        return np.diag(np.sum(self.A, axis=1))

    def _compute_normalized_laplacian(self) -> np.ndarray:
        """公式4：L̂ = I - D^{-1/2}AD^{-1/2}"""
        D_inv_sqrt = np.diag(1 / np.sqrt(np.diag(self.D) + 1e-8))  # 数值稳定性处理
        return np.eye(self.N) - D_inv_sqrt @ self.A @ D_inv_sqrt


def formation_similarity_error(L_current: np.ndarray, L_desired: np.ndarray) -> float:
    """编队相似性误差度量（对应论文公式5）"""
    # ||L_current - L_desired||_F^2 = tr((L_current - L_desired)^T(L_current - L_desired))
    return np.trace((L_current - L_desired).T @ (L_current - L_desired))


def compute_gradient(graph: FormationGraph, L_desired: np.ndarray) -> np.ndarray:
    """梯度计算（对应论文公式6-8）"""
    N = graph.N
    delta_L = 2 * (graph.L - L_desired)  # ∂f_s/∂L̂（公式7第二项）

    gradients = np.zeros((N, 2))

    for i in range(N):
        grad_w = np.zeros(N)
        for j in range(N):
            if i != j:
                # 公式7第三项：∂L/∂w_ij = (∂A/∂w_ij)
                mask = np.zeros((N, N))
                mask[i, j] = mask[j, i] = 1
                partial_L = mask
                grad_w[j] = np.trace(delta_L.T @ partial_L)  # 公式7

        # 公式8：∂w_i/∂p_i = 2(p_i - p_j)
        pos_diff = graph.positions[i] - graph.positions
        grad_pos = 2 * pos_diff

        # 链式法则：∂f_s/∂p_i = ∂f_s/∂w_i^T · ∂w_i/∂p_i（公式6）
        gradients[i] = grad_w @ grad_pos

    return gradients


import time  # 导入时间模块
# 实验参数配置（对应论文V.A节参数设置）
max_iter = 400  # 最大迭代次数（确保收敛）

total_move = 0
total_move_rec = []
time_rec = []
min_d_rec = []
cover_rec = []
uniform_rec = []
history = []

if __name__ == "__main__":
    # 记录开始时间
    start_time = time.time()
    # 生成期望编队（圆形布局，对应图3示例）
    img = cv2.imread('dragon.jpeg', cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (20, 20), interpolation=cv2.INTER_NEAREST)
    _, img_bin = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    data = img_bin.astype(bool)
    binary_data = np.where(data == 0, 1, 0)

    # 获取非零坐标 (行列格式)
    coords = np.argwhere(binary_data)


    # 计算几何中心
    center = np.mean(coords, axis=0)

    # 中心化坐标 (保持浮点精度)
    desired_positions = (coords - center) * 3.5 # 此处需要调整大小
    # temp = copy.deepcopy(desired_positions[:, 0])
    # desired_positions[:, 0] = desired_positions[:, 1]
    # desired_positions[:, 1] = temp
    desired_positions = np.array(desired_positions.tolist() + [[65, -10], [-90, 20.5], [90,15], [-98, -98], [-5, 5]], dtype=float)
    # 生成带噪声的初始位置（模拟实际环境扰动）
    print('长度: ', len(desired_positions))
    input()
    current_positions = settings.gen_settings(np.array([0, 0]), len(desired_positions)-5)[:, :2]
    current_positions = np.array(current_positions.tolist() + [[65, -10], [-90, 20.5], [90,15], [-98, -98], [-5, 5]], dtype=float)
    # 轨迹记录与优化过程（对应论文IV.C节优化流程）

    desired_graph = FormationGraph(desired_positions)
    L_desired = desired_graph.L

    for iter in range(max_iter):
        # check_collisions(current_positions)  # 实时碰撞检测

        graph = FormationGraph(current_positions)
        error = formation_similarity_error(graph.L, L_desired)

        print(f"=======iter {iter}=========")
        print('error: ', error)
        gradients = compute_gradient(graph, L_desired)

        # 计算机器人之间的避障斥力梯度
        def compute_inter_robot_gradient(current_positions, safe_distance=2, repulsion_coeff=1.0):
            """
            计算机器人之间的斥力梯度（只处理机器人间的避障）
            :param current_positions: 所有机器人位置 [N,2] 数组
            :param safe_distance: 触发斥力的最小间距（默认0.5）
            :param repulsion_coeff: 斥力强度系数（默认1.0）
            :return: 斥力梯度 [N,2] 数组
            """

            n_robots = current_positions.shape[0]
            obstacle_grad = np.zeros_like(current_positions)
            m_record = []

            # 遍历所有机器人对
            for i in range(n_robots):
                min_i = 1e10
                for j in range(n_robots):  # 避免重复计算
                    # 计算两机器人间距
                    if i == j:
                        continue
                    delta = current_positions[i] - current_positions[j]
                    dist = np.linalg.norm(delta)
                    if dist < min_i:
                        min_i = copy.deepcopy(dist)

                    if safe_distance > dist > 1e-6:  # 防止除零
                        # 斥力计算公式（方向为i远离j）
                        force = repulsion_coeff * (1 / dist - 1 / safe_distance) / dist * delta
                        obstacle_grad[i] += force
                        obstacle_grad[j] -= force  # 牛顿第三定律
                m_temp = copy.deepcopy(min_i)
                m_record.append(m_temp)
            print('m_record: ', min(m_record), max(m_record))
            u = np.std(m_record) / np.mean(m_record)

            s = np.linalg.norm(obstacle_grad)
            if s > 0.2:
                obstacle_grad = obstacle_grad / s * 0.2
            return obstacle_grad, min(m_record), u
        # 叠加避障梯度
        ob_grad, min_dist, u = compute_inter_robot_gradient(current_positions[:-5])
        c = covergence_test(current_positions[:-5], desired_positions[:-5])
        print('cover: ', c)
        print('uniform: ', u)
        print('min_dist: ', min_dist)
        cover_rec.append(copy.deepcopy(c))
        min_d_rec.append(copy.deepcopy(min_dist))
        uniform_rec.append(copy.deepcopy(u))
        print('编队:', np.max(abs(gradients)))
        # if iter < 300:
        gradients = gradients[:-5] * 0.04 + ob_grad  # 合并梯度并更新位置
        # else:
        #     gradients = gradients[:-5] * 0.1 + ob_grad  # 合并梯度并更新位置
        for i in range(len(gradients)):
            g = np.linalg.norm(gradients[i])
            if g > 1:
                gradients[i] = gradients[i] / g * 1
        current_positions[:-5] += gradients * 0.2
        current_time = time.time()
        time_rec.append(current_time-start_time)
        total_move += np.sum(np.linalg.norm(gradients * 0.4, axis=1))
        total_move_rec.append(total_move)
        history.append(current_positions[:-5].tolist())

    # 记录结束时间
    end_time = time.time()

    # 计算并打印总时间、优化时间、可视化时间
    total_time = end_time - start_time
    t = 'dragon'
    print(f"总运行时间: {total_time:.2f} s")
    with open(f'./data/run_data/sci_time_{t}.json', 'w+') as f:
        f.write(str(time_rec))
    with open(f'./data/run_data/sci_move_{t}.json', 'w+') as f:
        f.write(str(total_move_rec))
    with open(f'./data/run_data/sci_cover_{t}.json', 'w+') as f:
        f.write(str(cover_rec))
    with open(f'./data/run_data/sci_uniform_{t}.json', 'w+') as f:
        f.write(str(uniform_rec))
    with open(f'./data/run_data/sci_dist_{t}.json', 'w+') as f:
        f.write(str(min_d_rec))
    with open(f'./data/run_data/sci_pos_{t}.json', 'w+') as f:
        f.write(str(history))



    # # 可视化结果（对应论文图3可视化方法）
    plt.figure(figsize=(10, 6))

    # 绘制每条轨迹
    for drone_id in range(len(desired_positions)-5):
        # 轨迹线（虚线表示优化路径）
        plt.plot(
            [pos[drone_id][0] for pos in history],
            [pos[drone_id][1] for pos in history],
            linestyle='-',
            linewidth=0.2,
            alpha=0.4
        )

        # 初始位置（实心圆点）
        plt.scatter(
            history[-1][drone_id][0], history[-1][drone_id][1], s=60,
            edgecolor='k', linewidths=1,
            marker='o'
        )


    # 绘制期望位置（黑色星号）
    plt.scatter(
        desired_positions[:, 0], desired_positions[:, 1],
        marker='*', s=100, c='black',
        linewidths=1, edgecolors='k',
    )

    plt.grid(True, linewidth=0.3)
    plt.gca().set_aspect('equal')
    plt.show()