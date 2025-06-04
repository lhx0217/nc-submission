# This is a sample Python script.
import math
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import numpy as np


def optimal_assignment(pos_rec, desire_pos):
    # 计算欧几里得距离矩阵，形状为 (N, N)，每个元素表示一个点之间的距离
    cost_matrix = cdist(pos_rec, desire_pos, metric='euclidean')

    # 使用匈牙利算法（线性和分配问题）找到最优匹配
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # 根据最优匹配调整 desire_pos 的顺序
    optimal_desire_pos = desire_pos[col_ind]

    return optimal_desire_pos, row_ind, col_ind

def arrange_points_in_square(n):
    # 计算正方形的边长
    side_length = math.ceil(math.sqrt(n))
    if side_length * (side_length - 1) >= n:
        rows = side_length
        cols = side_length - 1
    else:
        rows = side_length
        cols = side_length
    # 生成每个点的坐标
    points = []
    for row in range(rows):
        for col in range(cols):
            points.append(((col - cols / 2) * 3, (row - rows / 2) * 3))
    points = points[:n]
    points.append([0, 0])
    # 返回点的坐标
    return np.array(points)


def checkConnected(adj_matrix):
    num_nodes = adj_matrix.shape[0]
    visited = np.zeros(num_nodes, dtype=bool)  # 初始化访问标记数组

    # 检查是否有节点未连接（即行全为0）
    for i in range(num_nodes):
        if np.all(adj_matrix[i, :] == 0):
            return False  # 如果发现未连接的节点，返回False

    # 从第一个节点开始DFS
    stack = [0]  # 使用列表作为堆栈存储待访问节点
    while stack:
        node = stack.pop()
        visited[node] = True  # 标记为已访问

        # 遍历当前节点的所有邻居
        for neighbor in range(num_nodes):
            if adj_matrix[node, neighbor] == 1 and not visited[neighbor]:
                stack.append(neighbor)  # 将未访问的邻居加入堆栈

    # 检查所有节点是否都被访问过
    return all(visited)


def generate_topology(num_students):
    # 生成随机邻接强度矩阵
    link_strength = np.random.rand(num_students, num_students)

    # 确保生成的邻接矩阵是对称的
    adj_matrix = (link_strength + link_strength.T) / 2

    threshold_initial = 0.9
    threshold_step = -0.01

    for threshold in np.arange(threshold_initial, 0, threshold_step):
        adj_matrix = np.zeros((num_students, num_students))  # 初始化邻接矩阵
        adj_matrix[link_strength >= threshold] = 1

        # 设置对角线元素为0
        np.fill_diagonal(adj_matrix, 0)

        # 检查连通性
        is_connected = checkConnected(adj_matrix)
        if is_connected == 1:  # 如果只有一个连通分量，图是连通的
            print(threshold)
            break

    return adj_matrix


def generate_unique_coordinates(N):
    # 生成随机坐标并确保唯一性
    coords = np.random.uniform(-20, 20, (N + 1, 2))
    coords[N, :] = np.zeros(2)
    return coords

def consensus_square(N):
    dt = 0.02
    pos_rec = generate_unique_coordinates(N)  # 初始位置
    vel_rec = np.zeros((N, 2))  # 初始速度
    desire_pos = arrange_points_in_square(N)  # 目标位置
    print("Desired positions:", desire_pos)
    adj_matrix = generate_topology(N + 1)  # 邻接矩阵
    print("Adjacency matrix:", adj_matrix)
    desire_pos, row_ind, col_ind = optimal_assignment(pos_rec, desire_pos)
    # 设置绘图
    # plt.ion()  # 打开交互模式
    fig, ax = plt.subplots()
    scatter = ax.scatter(pos_rec[:-1][:, 0], pos_rec[:-1][:, 1])  # 绘制初始位置
    ax.scatter([0], [0], c='red')
    ax.set_xlim([-30, 30])
    ax.set_ylim([-30, 30])
    ax.set_title('Consensus Process')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    for _ in range(10000):
        # 计算每个点的速度并更新位置
        for ind in range(N):
            indices = np.where(adj_matrix[ind, :] == 1)[0]
            control = np.zeros(2)
            for nei in indices:
                control += (desire_pos[ind] - desire_pos[nei]) - (pos_rec[ind] - pos_rec[nei])
            distances = np.linalg.norm(pos_rec[:-1] - pos_rec[ind], axis=1)
            # assert min(distances) < 1, "collide!!"

            ob = np.where((distances > 0) & (distances < 1.5))[0]
            cross_product = np.zeros(2)
            if len(ob):
                print('最小距离: ', min(distances[ob]))
                distances = distances[ob]
                # input(ob)
                weigh = 1.5 / distances - 1
                vec = (pos_rec[ind] - pos_rec[ob]) / distances.reshape(-1, 1)
                cmd_set = np.sum(vec * weigh.reshape(-1, 1), axis=0) / np.sum(weigh)
                cross_product = np.array([-cmd_set[1], cmd_set[0]]) + cmd_set
            control /= len(indices)
            l_control = np.linalg.norm(control)
            if l_control > 3:
                control = control / l_control * 3
            elif l_control < 0.05:
                control = control / l_control * 0.05
            vel_rec[ind] = control + cross_product

        # 更新位置
        for ind in range(N):
            pos_rec[ind] += vel_rec[ind] * dt

        # 更新图像
        scatter.set_offsets(pos_rec)  # 更新散点图的位置
        plt.pause(0.01)  # 暂停以显示图像并让其更新

    plt.ioff()  # 关闭交互模式
    plt.show()  # 显示最终图像



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    consensus_square(200)
