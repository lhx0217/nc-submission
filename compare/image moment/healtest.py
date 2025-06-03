import copy
import math
import time

import cv2
import numpy as np
from numpy import mean
import matplotlib.pyplot as plt
import scipy
from scipy.spatial import KDTree
import settings

total_move = 0
total_move_rec = []
min_dis_rec = []
time_rec = []
pos_rec = []
enter_rate = []
cover_rate = []
uniform_rate = []
start_time = time.time()

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


def entering_test(pos):
    global target
    min_dis = min_distance(pos, target)

    # 初始化目标点是否被覆盖的数组
    entered = np.zeros(len(pos), dtype=bool)

    # 从距离矩阵中选择最近的目标点，并分配无人机
    for i in range(len(pos)):
        if min_dis[i] <= 3:
            entered[i] = True

    # 计算覆盖率
    entering_rate = np.sum(entered) / len(pos)

    return entering_rate

def covergence_test(pos):
    global target
    # 初始化目标点是否被覆盖的数组
    covered = np.zeros(len(target), dtype=bool)
    min_dis = min_distance(target, pos)
    # 从距离矩阵中选择最近的目标点，并分配无人机
    for i in range(len(target)):
        if min_dis[i] <= 2:
            covered[i] = True
    # 计算覆盖率
    coverage_rate = np.sum(covered) / len(target)
    return coverage_rate

def uniform_test(pos):
    record_min_dis = []
    for index in range(len(pos)):
        pos_ = copy.deepcopy(pos.tolist())
        pos_.pop(index)
        min_dis = min_distance([pos[index]], pos_)
        record_min_dis.append(min_dis)
    uniform = np.std(record_min_dis)
    return uniform

def compute_legendre_and_derivatives(max_order, x):
    """计算x的Legendre多项式及其导数，p从0到max_order"""
    P = [np.zeros_like(x) for _ in range(max_order + 1)]
    dP = [np.zeros_like(x) for _ in range(max_order + 1)]
    P[0] = np.ones_like(x)
    dP[0] = np.zeros_like(x)
    if max_order >= 1:
        P[1] = x
        dP[1] = np.ones_like(x)
    for n in range(2, max_order + 1):
        P[n] = ((2 * n - 1) / n) * x * P[n - 1] - ((n - 1) / n) * P[n - 2]
        dP[n] = ((2 * n - 1) / n) * (P[n - 1] + x * dP[n - 1]) - ((n - 1) / n) * dP[n - 2]
    return P, dP


def compute_legendre_moments_robots(positions, max_order):
    """计算机器人群体的Legendre矩"""
    moments = {}
    x = positions[:, 0]
    y = positions[:, 1]

    P_x, _ = compute_legendre_and_derivatives(max_order, x)
    P_y, _ = compute_legendre_and_derivatives(max_order, y)

    for p in range(max_order + 1):
        for q in range(max_order + 1 - p):
            coeff = (2 * p + 1) * (2 * q + 1) / 4.0
            sum_pq = np.sum(P_x[p] * P_y[q])
            moments[(p, q)] = coeff * sum_pq
    return moments


def compute_legendre_moments(image, max_order):
    height, width = image.shape
    x = np.linspace(-1, 1, width)
    y = np.linspace(-1, 1, height)

    # 预计算多项式矩阵
    P_x = legendre_polynomials(max_order, x)  # (max_order+1, width)
    P_y = legendre_polynomials(max_order, y)  # (max_order+1, height)

    # 矩阵运算优化
    moments = {}
    for p in range(max_order + 1):
        for q in range(max_order + 1 - p):
            coeff = (2 * p + 1) * (2 * q + 1) / 4
            moment = coeff * (P_y[q] @ image @ P_x[p].T)
            moments[(p, q)] = moment
    return moments


def legendre_polynomials(max_order, x):
    P = [np.zeros_like(x) for _ in range(max_order + 1)]
    P[0] = np.ones_like(x)
    if max_order >= 1:
        P[1] = x

    for n in range(2, max_order + 1):
        P[n] = ((2 * n - 1) / n * x * P[n - 1]) - ((n - 1) / n * P[n - 2])
    return P


def legendre_polynomials_(max_order, x):
    """迭代法预计算所有阶Legendre多项式"""
    P = [np.zeros_like(x) for _ in range(max_order + 1)]
    P[0] = np.ones_like(x)
    if max_order >= 1:
        P[1] = x

    for n in range(2, max_order + 1):
        P[n] = ((2 * n - 1) / n * x * P[n - 1]) - ((n - 1) / n * P[n - 2])
    return P


def reconstruct_from_moments(moments, max_order, grid_size=41):
    """从Legendre矩重建图像（优化版）"""
    x = np.linspace(-1, 1, grid_size)
    y = np.linspace(-1, 1, grid_size)
    X, Y = np.meshgrid(x, y)

    # 预计算所有阶多项式（向量化）
    P_x = legendre_polynomials_(max_order, x)  # x方向多项式
    P_y = legendre_polynomials_(max_order, y)  # y方向多项式

    # 初始化重建图像
    reconstructed = np.zeros_like(X)

    # 按正交基展开公式重建
    for (p, q), M_pq in moments.items():
        if p + q > max_order:
            continue
        # 外积计算基函数分量
        basis = np.outer(P_y[q], P_x[p])
        reconstructed += M_pq * basis

    return reconstructed


def global_gradient_update(s, M_target, max_order, dt):
    """带梯度计算的全局更新"""
    global scale
    scale = 50
    s = s / scale
    M_current = compute_legendre_moments_robots(s, max_order)
    E = {(p, q): M_current.get((p, q), 0) - M_target.get((p, q), 0)
         for (p, q) in M_target.keys()}

    x = s[:, 0]
    y = s[:, 1]
    P_x, dP_x = compute_legendre_and_derivatives(max_order, x)
    P_y, dP_y = compute_legendre_and_derivatives(max_order, y)

    gradients = np.zeros_like(s)
    for i in range(len(s)):
        delta_x, delta_y = 0.0, 0.0
        for (p, q) in M_target.keys():
            d = p + q
            gamma = (d+1) ** (-1.7)  # 论文建议的增益系数
            coeff = (2 * p + 1) * (2 * q + 1) / 4.0
            dM_dx = coeff * dP_x[p][i] * P_y[q][i]
            dM_dy = coeff * P_x[p][i] * dP_y[q][i]
            error = E[(p, q)]
            delta_x += gamma * error * dM_dx
            delta_y += gamma * error * dM_dy
        gradients[i] = [-delta_x, -delta_y]

    speeds = []

    for robot in s * scale:
        distance, idx = tree.query(robot)
        closest = target[idx]

        dx = closest[0] - robot[0]
        dy = closest[1] - robot[1]

        if distance > 3:
            norm = math.hypot(dx, dy)
            if norm > 1e-6:  # 避免除以零
                factor = 0.2 / norm
                vx, vy = dx * factor, dy * factor
            else:
                vx, vy = 0, 0
        else:
            vx, vy = 0, 0

        speeds.append((vx, vy))


        # 计算机器人之间的避障斥力梯度
    def compute_inter_robot_gradient(current_positions, safe_distance=1.5, repulsion_coeff=1.0):
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
            min_i = 100000.0

            for j in range(n_robots):  # 避免重复计算
                if i == j:
                    continue
                # 计算两机器人间距
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
        u = np.std(m_record) / np.mean(m_record)
        s = np.linalg.norm(obstacle_grad)
        if s > 0.5:
            obstacle_grad = obstacle_grad / s * 0.3
        return obstacle_grad, min(m_record), u

    ob_grad, min_dist, u = compute_inter_robot_gradient(s*scale)
    gradients = gradients * dt
    for i in range(len(gradients)):
        g = np.linalg.norm(gradients[i])
        if g > 2:
            gradients[i] /= g * 0.5
    print(np.max(gradients))
    s_new = s*scale + gradients + ob_grad + np.array(speeds)
    global total_move
    total_move += np.sum(np.linalg.norm(gradients + ob_grad + np.array(speeds), axis=1))
    total_move_rec.append(total_move.tolist())
    current_time = time.time()
    time_rec.append(current_time - start_time)
    cp = (s * scale).tolist()
    pos_rec.append(cp)
    min_dis_rec.append(min_dist)
    e = entering_test(s * scale)
    c = covergence_test(s * scale)
    print('cover: ', c)
    print('uniform: ', u)
    enter_rate.append(e)
    cover_rate.append(c)
    u_temp = copy.deepcopy(u)
    uniform_rate.append(u_temp)
    return np.clip(s_new, -scale, scale), np.sum(np.array(list(E.values())) ** 2)


def plot_comparison(target_image, robot_positions, M_target, max_order):
    """绘制四组对比图"""
    plt.figure(figsize=(12, 10))

    # 原始目标图像
    plt.subplot(2, 2, 1)
    plt.scatter(target_image[:, 0], target_image[:, 1], s=25, alpha=0.6)
    plt.title("Original Target Image")
    plt.axis('off')

    # 目标矩重建图像
    plt.subplot(2, 2, 2)
    target_reconstructed = reconstruct_from_moments(M_target, max_order)
    plt.imshow(target_reconstructed, cmap='viridis', extent=[-50, 50, -50, 50])
    plt.title("Target Reconstruction from Moments")
    plt.axis('off')

    # 机器人位置分布
    plt.subplot(2, 2, 3)
    plt.scatter(robot_positions[:, 0], robot_positions[:, 1], s=25, alpha=0.6)
    plt.title("Robot Positions")
    plt.grid(True)

    # 机器人矩重建图像
    plt.subplot(2, 2, 4)
    M_robot = compute_legendre_moments_robots(robot_positions / scale, max_order)
    robot_reconstructed = reconstruct_from_moments(M_robot, max_order)
    plt.imshow(robot_reconstructed, cmap='viridis', extent=[-50, 50, -50, 50])
    plt.title("Reconstruction from Robot Moments")
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def read_gray_mtr():
    """
    读取包含gray_mtr变量的MAT文件
    参数：
        file_path : 文件路径
        squeeze   : 是否去除单维度（默认True）
    返回：
        gray_mtr  : 灰度矩阵数据（numpy数组）
    """
    # file_path = 'star.mat'
    # mat_data = scipy.io.loadmat(file_path)
    #
    # # 处理MATLAB与numpy的维度存储差异
    # data = np.transpose(mat_data['gray_mtr'])
    # print(data.shape)
    img = cv2.imread('../../models/dragon.jpeg', cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (18, 18), interpolation=cv2.INTER_NEAREST)
    _, img_bin = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    data = img_bin.astype(bool)
    binary_data = np.where(data == 0, 1, 0)
    s = np.sum(binary_data)
    # 获取非零坐标 (行列格式)
    coords = np.argwhere(binary_data)

    if len(coords) == 0:
        return np.array([])  # 返回空数组表示无坐标

    # 计算几何中心
    center = np.mean(coords, axis=0)

    # 中心化坐标 (保持浮点精度)
    centered_coords = coords - center

    # 将坐标格式转换为 (x,y) 坐标系（列坐标为x，行坐标为y）
    centered_coords_3d = np.hstack((
        centered_coords[:, [1, 0]],  # 保持x,y坐标转换
        np.zeros((len(centered_coords), 1))  # 新增z维度
    )) / 22 * 80  # 此处需要调整大小
    return centered_coords_3d

if __name__ == "__main__":
    global target
    max_order = 30
    num_robots = 200
    num_iterations = 400

    # 生成目标图像（bunny head示例）
    # 加载目标图像
    # img = cv2.imread('../mouse.jpg', cv2.IMREAD_GRAYSCALE)
    # img = cv2.resize(img, (63, 63), interpolation=cv2.INTER_NEAREST)
    # _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    # target_image = img.astype(bool)
    # l = np.sum(target_image)

    # target = image_to_points(target_image)
    # # 计算目标矩（使用图像矩）
    # M_target = compute_legendre_moments(target_image, max_order)
    s = (settings.gen_settings(np.array([0, 0]), num_robots))[:, :2]
    # 初始化机器人位置
    # s = np.random.uniform(-1, 1, (num_robots, 2))
    target = read_gray_mtr()[:, :2]
    global tree
    tree = KDTree(target)
    M_target = compute_legendre_moments_robots(target / 50, max_order)
    # 运行梯度下降
    errors = []
    for iter in range(num_iterations):
        s, error = global_gradient_update(s, M_target, max_order, dt=0.0004)
        errors.append(error)
        print(f'============={iter}==============')
        print('error: ', error)
    t = 'dragon'
    with open(f'../../data/run_data/heal_time_{t}.json', 'w+') as f:
        f.write(str(time_rec))
    with open(f'../../data/run_data/heal_move_{t}.json', 'w+') as f:
        f.write(str(total_move_rec))
    with open(f'../../data/run_data/heal_dist_{t}.json', 'w+') as f:
        f.write(str(min_dis_rec))
    with open(f'../../data/run_data/heal_enter_{t}.json', 'w+') as f:
        f.write(str(enter_rate))
    with open(f'../../data/run_data/heal_cover_{t}.json', 'w+') as f:
        f.write(str(cover_rate))
    with open(f'../../data/run_data/heal_uniform_{t}.json', 'w+') as f:
        f.write(str(uniform_rate))
    with open(f'../../data/run_data/heal_pos_{t}.json', 'w+') as f:
        f.write(str(pos_rec))
    # 绘制结果对比
    plot_comparison(target, s, M_target, max_order)

    # 绘制误差曲线
    plt.figure()
    plt.plot(errors)
    plt.xlabel('Iteration')
    plt.ylabel('Moment Error (2-norm)')
    plt.title('Convergence of Legendre Moments')
    plt.show()