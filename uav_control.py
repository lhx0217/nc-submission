# I will now rewrite the entire `uav_control.py` based on the user's complete code,
# adding detailed English comments to all classes, functions, and key logic blocks.

import copy
import numpy as np
from collections import Counter
from scipy.spatial import KDTree


class UAV:
    """
    UAV class simulating an individual drone in a swarm.
    Implements shape-aware control based on entering, exploration, and interaction strategies.
    """

    def __init__(self, num, sim, **kwargs):
        self.num = num  # Unique ID of the UAV
        self.sim = sim  # Simulation environment reference

        # Load initial shape data (centered at origin) and compute its KD-tree
        self.init_graph = np.array(kwargs['graph'], dtype=float)
        self.graph_center = np.array(kwargs['des_pos'])
        self.graph = KDTree(copy.deepcopy(self.init_graph) + self.graph_center)
        self.graph_color = np.array(kwargs['graph_color'], dtype=float)
        self.graph_angle = np.array(kwargs['des_angle'])
        self.graph_vel = kwargs['graph_vel']
        self.graph_angle_vel = kwargs['angle_vel']

        # UAV motion state
        self.pos = self.get_pos()
        self.vel = np.array([0, 0, 0], dtype=float)
        self.pos_color = 0  # 0: white (not in shape), 1: black (in shape)

        # Neighborhood sensing and communication buffers
        self.kmeans = kwargs['kmeans']
        self.neigh_set = []
        self.neigh_vel = []
        self.neigh_contain = []
        self.neigh_avg = []
        self.neigh_avg_dist = []
        self.nei_len = []
        self.neigh_indexing = []
        self.nei_color = []

        # Shape structure association
        self.contain = 0  # Number of shape grid points this UAV is responsible for

        # Sensing and control parameters
        self.t = kwargs['t']
        self.t1, self.t2 = 0, 0
        self.r_sense = kwargs['r_sense']
        self.r_avoid = kwargs['r_avoid']
        self.o_avoid = kwargs['r_avoid']

        # Buffers used across functions
        self.indices_within_distance = None
        self.points_within_distance = None
        self.min_index = None

        # Performance metrics
        self.cmd_set = 0
        self.contain_exp = 0
        self.length_finetuned = 0
        self.enter = 0
        self.interact = 0
        self.explore = 0
        self.total_control = 0
        self.min_dis_n = 3

        # Recording
        self.record_pos = []
        self.record_target = []
        self.record_nei_std = []
        self.round = 0
        self.nei_index_0 = []

    def get_pos(self):
        """Reset control metrics and return current UAV position."""
        self.contain_exp = 0
        self.length_finetuned = 0
        self.enter = 0
        self.interact = 0
        return self.sim.pos[self.num]

    def add_noise(self, vec, sigma=0.05, delta=0.1):
        """Add Gaussian and uniform noise to a vector."""
        return vec + np.random.normal(0, sigma, size=vec.shape) + np.random.uniform(-delta, delta, size=vec.shape)

    def unit_vector(self, start, end):
        """Return the unit direction vector from start to end."""
        dx, dy, dz = end[0] - start[0], end[1] - start[1], end[2] - start[2]
        mag = (dx ** 2 + dy ** 2 + dz ** 2) ** 0.5
        return np.array([dx / mag, dy / mag, dz / mag], dtype=float) if mag else np.array([0.0, 0.0, 0.0])

    def find_closest_points(self, points, target, n):
        """Return the indices of the n closest points to a target."""
        distances = np.array([np.linalg.norm(p - target) for p in points])
        return np.argsort(distances)[:n]

    def weight_function(self, x, r):
        """Cosine-based weight function used for soft attraction/repulsion control."""
        y = (np.cos(np.pi * x / r) + 1) / 2
        y[x <= 0] = 1
        y[x > r] = 0
        return y

    def get_graph_pos_angle(self):
        """Retrieve dynamic shape center and angular velocity."""
        return self.sim.get_graph_pos_angle_vel()

    def input_container(self, index, num):
        """Register the number of grid points this UAV is assigned in the simulation."""
        self.sim.input_container(index, num)

    def GetNeighborSet(self):
        """Query and update neighbor data from the simulation."""
        self.obstacles = None
        (self.neigh_set, self.neigh_vel, self.neigh_contain, self.neigh_avg_contain,
         self.neigh_avg, self.neigh_avg_dist, self.nei_len, self.nei_color) = self.sim.get_neighbor(self.num)
        self.neigh_set = self.add_noise(self.neigh_set)
        if len(self.neigh_set):
            self.min_dis_n = np.min(np.linalg.norm(self.pos - self.neigh_set, axis=1))
        else:
            self.min_dis_n = self.r_sense

    def get_pos_color(self):
        """Determine whether UAV is inside the black region of the shape."""
        distance, self.min_index = self.graph.query(self.pos, k=1)
        self.pos_color = 1 if distance <= 1 else 0
        self.sim.pos_color[self.num] = self.pos_color
        ind = self.graph.query_ball_point(self.pos, self.r_sense)
        self.indices_within_distance = np.array(ind)
        if len(self.indices_within_distance):
            self.points_within_distance = self.graph.data[self.indices_within_distance]
        else:
            self.points_within_distance = np.array([])

    def find_closer_points_large_data(self, target_points, neighbor_points, point_A):
        """Find shape grid points closer to current UAV than to any neighbor."""
        if not np.size(neighbor_points) or not np.size(target_points):
            return [], -1, None, None
        target_points_t = KDTree(neighbor_points)
        distances_from_A = np.linalg.norm(target_points - point_A, axis=1)
        min_distances_from_neighbors, min_index = target_points_t.query(target_points)
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
        dis, closer_nei_indices = zip(*sorted_pairs)
        kmeans = target_points[closer_points_indices]
        return kmeans, closer_points_indices, list(closer_nei_indices), [counts.get(k, 0) for k in closer_nei_indices]

    def EnteringCmd(self):
        """Compute velocity vector guiding UAV into the black region of the target shape."""
        if self.pos_color:
            if not np.size(self.neigh_vel):
                self.enter = 0
                return np.zeros(3)
            return np.zeros(3)
        enter = self.graph.data[self.min_index] - self.pos
        self.enter = np.linalg.norm(enter)
        if self.enter > 2:
            enter = self.unit_vector([0, 0, 0], enter) * 2
        self.enter = np.linalg.norm(enter)
        return enter

        # Continuing full implementation of `ExplorationCmd`, `InteractionCmd_n`, `op_uav_cmd`, and `run` methods

    def ExplorationCmd(self, Kmeans=None, points_index=None, neigh_indexing0=None, edge_length=None):
        """Compute exploration vector using local neighborhood structure and shape distribution."""
        indices_within_distance = self.indices_within_distance
        points_within_distance = self.points_within_distance
        if not np.size(points_within_distance) or not len(self.neigh_set):
            return np.array([0, 0, 0], dtype=float)

        avg_control = np.zeros(3)
        avg_explore = np.zeros(3)
        goal_set_neigh = np.zeros(3)

        if not neigh_indexing0:
            neigh_indexing0 = []
            edge_length = []

        self.nei_index_0 = neigh_indexing0
        if neigh_indexing0:
            points_within_distance, indices_within_distance = Kmeans, indices_within_distance[points_index]
            self.points_within_distance, self.indices_within_distance = points_within_distance, indices_within_distance
            neigh_indexing = np.array(neigh_indexing0[:6])
            self.neigh_indexing = neigh_indexing
            avg_des0 = np.mean(np.linalg.norm(self.pos - self.neigh_set[neigh_indexing], axis=1))
            avg_des = (sum(self.neigh_avg) + avg_des0) / (len(self.neigh_avg_dist) + 1)
            self.sim.avg_dist[self.num] = avg_des0
            self.sim.nei_len[self.num] = len(neigh_indexing)
            self.sim.avg_des[self.num] = avg_des0 if self.round == 0 else avg_des

            self.record_nei_std = []
            for nei in self.neigh_set[neigh_indexing]:
                self.record_nei_std.append(np.linalg.norm(self.pos - nei))
                avg_control += (np.linalg.norm(self.pos - nei) - avg_des) * self.unit_vector(self.pos, nei)

            for nei, nei_dis, color in zip(self.neigh_set, self.neigh_avg_dist, self.nei_color):
                if color:
                    avg_explore += (nei_dis - avg_des0) * self.unit_vector(self.pos, nei)

            self.contain = len(Kmeans)
            self.input_container(self.num, self.contain)

            if edge_length:
                edge_length = edge_length[:6]
                for nei, edge in zip(neigh_indexing, edge_length):
                    goal_set_neigh += edge * (self.neigh_contain[nei] - self.contain) * self.unit_vector(self.pos,
                                                                                                         self.neigh_set[
                                                                                                             nei])
                if len(neigh_indexing) and sum(edge_length):
                    goal_set_neigh /= len(neigh_indexing)
                    goal_set_neigh /= sum(edge_length)

                if self.round == 0:
                    self.sim.avg_contain[self.num] = self.contain
                    goal_set_neigh = np.zeros(3)
                else:
                    self.sim.avg_contain[self.num] = (np.sum(
                        self.neigh_avg_contain[neigh_indexing]) + self.contain) / (len(neigh_indexing) + 1)
                    if np.linalg.norm(goal_set_neigh) > 1:
                        goal_set_neigh = self.unit_vector(np.zeros(3), goal_set_neigh)

        if not len(points_within_distance):
            self.length_finetuned = np.linalg.norm((avg_explore + avg_control))
            self.contain_exp = np.linalg.norm(goal_set_neigh)
            return (avg_explore + avg_control) + goal_set_neigh

        self.record_pos.append(copy.deepcopy(list(self.pos)))
        vec_within_distance = self.pos - points_within_distance
        dis = np.linalg.norm(vec_within_distance, axis=1)
        distances_neigh = np.linalg.norm(points_within_distance[:, None, :] - self.neigh_set, axis=2)
        inside_any_sphere = np.any(distances_neigh < 0.2, axis=1)
        points_without_occupy = points_within_distance[~inside_any_sphere]
        if not len(points_without_occupy):
            self.length_finetuned = np.linalg.norm((avg_explore + avg_control) * 3)
            self.contain_exp = np.linalg.norm(goal_set_neigh)
            return (avg_explore + avg_control) * 3 + goal_set_neigh

        weigh = self.weight_function(dis[~inside_any_sphere], self.r_sense)
        goal_set_expl = np.sum(points_without_occupy * weigh[:, None], axis=0) / sum(weigh)
        cmd_set = (goal_set_expl - self.pos)
        self.contain_exp = np.linalg.norm(goal_set_neigh * 2)
        self.length_finetuned = np.linalg.norm((3 * avg_control + avg_explore / 2))
        self.explore = np.linalg.norm(cmd_set * 2)
        return cmd_set * 2 + (avg_control * 3 + avg_explore / 2) + goal_set_neigh * 2

    def InteractionCmd_n(self, r_avoid):
        """Compute repulsion command based on nearby neighbors within r_avoid distance."""
        if not len(self.neigh_set):
            return np.zeros(3)
        distances = np.linalg.norm(self.neigh_set - self.pos, axis=1)
        indices_within_distance = np.where(distances < r_avoid)[0]
        if not len(indices_within_distance):
            return np.zeros(3)
        points_within_ravoid = self.neigh_set[indices_within_distance]
        weights = r_avoid / distances[indices_within_distance] - 1
        vectors = (self.pos - points_within_ravoid) / distances[indices_within_distance].reshape(-1, 1)
        return np.sum(vectors * weights[:, None], axis=0) / sum(weights)

    def op_uav_cmd(self, cmd_set, num):
        """Apply computed command vector to update UAV position and velocity."""
        if self.round == 0:
            return
        vel = self.sim.vel[num] * 0.3 + cmd_set * 0.7
        pos = copy.deepcopy(self.sim.pos[num]) + vel * self.t
        self.sim.total_move += np.linalg.norm(pos - self.sim.pos[num])
        self.sim.pos[num] = pos
        self.vel = vel
        self.sim.vel[num] = vel

    def run(self, round):
        """Main per-timestep control loop for each UAV."""
        print(f"=============== ROUND {round} | UAV {self.num} ===============")
        self.round = round
        self.pos = self.add_noise(self.get_pos())
        self.graph, self.graph_vel = self.get_graph_pos_angle()
        if round == 1 and self.num == 0:
            with open('./data/graph.json', 'w+') as f:
                f.write(str(self.graph.data.tolist()))
        self.graph_color = np.ones(len(self.graph.data))
        self.get_pos_color()
        self.GetNeighborSet()

        cmd_enter = self.EnteringCmd()
        Kmeans, points_index, neigh_indexing0, edge_length = self.find_closer_points_large_data(
            self.points_within_distance, self.neigh_set, self.pos)
        cmd_explore = self.ExplorationCmd(Kmeans, points_index, neigh_indexing0, edge_length)
        cmd_interact = self.InteractionCmd_n(self.r_avoid)

        # Combine control strategies with dampened response if interaction dominates
        scale = np.max([0, 1 - np.linalg.norm(cmd_interact)])
        cmd_set = cmd_enter * scale + cmd_interact + cmd_explore * scale

        # Limit maximum velocity
        self.cmd_set = np.linalg.norm(cmd_set)
        v = 2
        if len(self.neigh_set):
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
            cmd_set = self.unit_vector(np.zeros(3), cmd_set) * v

        self.interact = np.linalg.norm(cmd_interact)
        self.op_uav_cmd(cmd_set, self.num)
        self.cmd_set = np.linalg.norm(cmd_set)

