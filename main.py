import copy
import threading
import time
# import publish.pc_publish as pc
import uav_control
import graph_base
import matlab_test
import utils
import settings
import numpy as np
from scipy.spatial import KDTree
import airsim_control as airsim
from graph_base import *

graph = []
destination = [0, 0, 0]
graph_vel = np.array([0, 0, 0], dtype=float)
angle_vel = np.array([0, 0, 0], dtype=float)
rsense = 10
swarm_size = 200
graph_color = [1] * len(graph)
# pos = settings.gen_random_2d(swarm_size,  [-15, -15, 0], 5)
pos = settings.gen_settings(np.array([0, 0]), swarm_size)
init_state = {
    "r_sense": rsense,
    "pos": pos,
    "graph": graph,
    "des_pos": [0, 0, 0],
    "des_angle": [0, 0, 0],
    "graph_color": graph_color,
    "graph_vel": graph_vel,  # 单位m/s
    "angle_vel": angle_vel,  # 单位rad/s
    "destination": destination,
    "obstacles": [],
    "param": None
}
sim = airsim.AirSim(**init_state)
stop_flag = False
destination_arrival_status = False
record_rate = []
record_pos = []
record_comm = []
record_time = []

def run_uavs(uav_list, sim, init_state, rounds):
    global stop_flag
    start_time = time.time()
    print('rounds: ', rounds)
    temp_uav_list = uav_list
    for round in range(rounds):
        current_time = time.time()
        sim.GetDynFormation()
        for uav in temp_uav_list:
            uav.run(round)
        sim.current_time = init_state['t'] * round
        if round >= 0:
            rec, comm, center = matlab_test.all_test(sim, init_state, uav_list)
            record_rate.append(rec)
            record_pos.append(sim.pos.tolist())
            record_comm.append(comm)
            record_time.append(current_time - start_time)
        if not (round+1) % 5 and round > 0:
            p = threading.Thread(target=utils.file_write,
                                 args=(init_state['kmeans'], record_rate, record_pos, record_comm, record_time))
            p.start()

def drone_swarm_formation(graph, roun):
    global init_flag, stop_flag
    stop_flag = False
    if type(graph) is dict:
        graph = graph['graph']
    global sim
    sim.init_graph = graph
    # if not init_flag:
    #     return
    init_state = {
        "r_sense": rsense,
        "r_avoid": 1,  # 6/2
        "t": 0.05,
        "pos": sim.pos,
        "graph": graph,
        "des_pos": sim.graph_center,
        "des_angle": sim.graph_angle,
        "graph_color": [1] * len(graph),
        "kmeans": 1,
        "graph_vel": [0, 0, 0],  # 单位m/s
        "angle_vel": [0, 0, 0],  # 单位rad/s
        "destination": destination,
        "obstacles": [],
        "param": None
    }
    uav_list = []
    for num in range(swarm_size):
        uav = uav_control.UAV(num, sim, **init_state)
        uav_list.append(uav)
    # p = threading.Thread(target=run_uavs, args=(uav_list, sim, init_state))
    # p.start()
    init_flag = False
    run_uavs(uav_list, sim, init_state, roun)


def drone_swarm_stop():
    global stop_flag, init_flag, sim
    # 停止当前集群循环
    stop_flag = True
    # 初始化集群状态
    init_flag = True


def get_swarm_center(args=None):
    global sim
    return sim.graph_center



if __name__ == "__main__":
    # # Generate points for a square formati
    # graph = fibonacci_sphere(30, 4000)
    # graph = generate_points_in_sphere(radius=30,
    # center=(0, 0, 0),
    # num_r=10,
    # num_theta=35,
    # num_phi=100)
    graph = read_gray_mtr()
    # graph = read_stl('line_sphere.stl')
    # graph = read_stl('flower.stl')
    drone_swarm_formation(graph, 1000)
