import threading
import time
import random
import uav_control_ms
import math
import itertools
import matlab_test, utils
from graph_base import *
import airsim_control_ms as airsim
import settings
import numpy as np

def distance(p1, p2):
    """Compute the Euclidean distance between two 3D points."""
    return math.sqrt((p1[0] - p2[0])**2 +
                     (p1[1] - p2[1])**2 +
                     (p1[2] - p2[2])**2)

def closest_points(points):
    """Find the closest two 3D points in a list and return the minimum distance."""
    min_distance = float('inf')

    # Generate all pairs of points
    for p1, p2 in itertools.combinations(points, 2):
        dist = distance(p1, p2)
        if dist < min_distance:
            min_distance = dist
            closest_pair = (p1, p2)

    return min_distance


def fibonacci_sphere(radius, samples=1):
    """
    在球面上生成均匀分布的点，使用黄金分割比例。
    参数:
    - radius: 球面的半径。
    - samples: 要生成的点的数量。
    返回:
    一个形状为 (samples, 3) 的数组，包含生成的点的笛卡尔坐标。
    """
    points = []
    offset = 2.0 / samples
    increment = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(samples):
        y = ((i * offset) - 1) + (offset / 2)
        r = np.sqrt(1 - y ** 2)
        phi = ((i + 1) % samples) * increment
        x = np.cos(phi) * r * radius
        z = np.sin(phi) * r * radius
        points.append([x, y * radius, z])
    return np.array(points)

def matlab(points):
    # 将浮点坐标转换为整数
    points_int = np.round(points).astype(int)
    ne = []
    for points in points_int:
        pre = points
        print(f'Gray_image({pre[0] + 15},{pre[1] + 15},{pre[2] + 15})=0;')
        ne.append(pre)


destination_arrival_status = False
record_rate = []
record_comm = []
record_pos = []
record_time = []
def run_uavs(uav_list, init_state, sim, rounds=1000):
    start_time = time.time()
    temp_uav_list = uav_list
    for round in range(1000):
        current_time = time.time()
        for uav in temp_uav_list:
            uav.run(round)
        if round >= 0:
            rec, comm, no_enter = matlab_test.all_test(sim, init_state, uav_list)
            record_rate.append(rec)
            record_comm.append(comm)
            record_pos.append(sim.pos.tolist())
            record_time.append(current_time - start_time)
        if not (round + 1) % 5 and round > 0:
            p = threading.Thread(target=utils.file_write,
                                 args=(0, record_rate, record_pos, record_comm, record_time))
            p.start()


def run():
    graph = read_gray_mtr('../../models')
    graph_color = [1] * len(graph)
    pos_set = np.array(settings.gen_settings([0,0], 200))
    inform_list = [random.randint(0, len(pos_set)) for _ in range(int(len(pos_set) / 3))]
    r_avoid = 1.4
    init_state = {
        "l_grid": 1,
        "r_sense": 6,
        "r_avoid": r_avoid,
        "t": 0.05,
        "pos": pos_set,
        "graph": graph,
        "des_pos": [0, 0, 0],
        "des_angle": [0, 0, 0],
        "graph_color": graph_color,
        'kmeans': 0
    }
    sim = airsim.AirSim(**init_state)
    uav_list = []
    for num in range(len(pos_set)):
        is_inform = num in inform_list
        uav = uav_control_ms.UAV(num, 1 if is_inform else 0, sim, **init_state)
        uav_list.append(uav)
    run_uavs(uav_list, init_state, sim, rounds=1000)

if __name__ == "__main__":
    run()