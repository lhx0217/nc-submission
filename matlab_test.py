import copy
import numpy as np


def min_distance(A, B):
    """
    Compute the minimum Euclidean distance from each point in A to any point in B.

    Args:
        A (np.ndarray): Array of shape (n, d), representing n points.
        B (np.ndarray): Array of shape (m, d), representing m reference points.

    Returns:
        list: List of minimum distances for each point in A.
    """
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
    """
    Evaluate shape coverage by checking if target points are sufficiently close to UAVs.

    Args:
        airsim: Simulation state.
        m_l (float): Distance threshold for a point to be considered covered.

    Returns:
        float: Coverage rate (proportion of target points covered).
    """
    pos = copy.deepcopy(airsim.pos)
    target = airsim.graph

    covered = np.zeros(len(target), dtype=bool)
    min_dis = min_distance(target, pos)

    for i in range(len(target)):
        if min_dis[i] <= m_l + 0.2:
            covered[i] = True

    return np.sum(covered) / len(target)


def entering_test(airsim):
    """
    Test how many UAVs have entered the target shape (i.e., within 1.1 units of any point).

    Args:
        airsim: Simulation state.

    Returns:
        tuple: (entry_rate, list of indices of UAVs that failed to enter).
    """
    pos = copy.deepcopy(airsim.pos)
    target = airsim.graph
    min_dis = min_distance(pos, target)

    entered = np.zeros(len(pos), dtype=bool)
    for i in range(len(pos)):
        if min_dis[i] <= 1.1:
            entered[i] = True

    entering_rate = np.sum(entered) / len(pos)
    not_entered_indices = np.where(entered == False)[0]

    return entering_rate, not_entered_indices


def all_test(airsim, init_state, uav_list):
    """
    Collect multiple metrics to evaluate UAV performance during simulation.

    Args:
        airsim: Simulation environment object.
        init_state (dict): Simulation initialization parameters.
        uav_list (list): List of UAV objects.

    Returns:
        tuple:
            - result_metrics (list): Includes entering rate, coverage, movement, etc.
            - command_distribution (list): Normalized command vector contributions.
            - non_entered_uavs (list): UAV indices that failed to enter shape.
    """
    km = init_state['kmeans']
    move = airsim.total_move
    avg_des = np.std(airsim.avg_des) / np.mean(airsim.avg_des)
    std_contain = np.std(airsim.container) / np.mean(airsim.container)
    std_dist = []

    command_percentage = [0] * 8
    interact, vel, min_dist = [], [], []

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

    std_dist2 = np.mean(airsim.avg_dist)  # Approximate shape uniformity indicator

    e, non_enter = entering_test(airsim)
    c = covergence_test(airsim, np.mean(std_dist))
    u = np.std(min_dist) / np.mean(min_dist)

    result_metrics = [
        e,                     # Entering rate
        c,                     # Convergence coverage rate
        u,                     # Distance uniformity
        move,                  # Total movement
        avg_des,               # Angle deviation
        np.mean(vel),          # Average UAV velocity
        std_dist2,             # Avg. neighbor distance
        std_contain,           # Container variance
        np.min(min_dist),      # Closest distance observed
        uav_list[0].round      # Simulation round
    ]

    return result_metrics, command_percentage, non_enter
