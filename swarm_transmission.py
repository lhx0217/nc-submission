# Outputting the complete content of swarm_transmission.py with all functionality and full English comments

import numpy as np
from scipy.spatial import KDTree


class AirSim:
    """
    Simulation class emulating an AirSim-like environment for UAV swarm behavior.
    It provides dynamic formation transformations, neighbor querying, and
    records positional and control statistics for each UAV.
    """

    def __init__(self, **kwargs):
        # UAV initialization
        self.init_pos = np.array(kwargs['pos'], dtype=float)  # Initial positions
        self.num = len(kwargs['pos'])  # Number of UAVs
        self.pos = np.array(kwargs['pos'], dtype=float)       # Current positions
        self.vel = np.zeros((self.num, 3), dtype=float)        # Velocities (initialized to zero)
        self.r_sense = kwargs['r_sense']                       # Sensing range

        # Target shape representation
        self.init_graph = kwargs['graph']                      # Initial graph at origin
        self.graph = kwargs['graph']                           # Current transformed graph
        self.kd_graph = None                                   # KDTree for nearest neighbor queries

        # Environment configuration
        self.obstacles = kwargs['obstacles']                   # Obstacles (currently unused)
        self.graph_center = np.array(kwargs['des_pos'], dtype=float)  # Center of the shape
        self.graph_vel = np.array(kwargs['graph_vel'], dtype=float)   # Target velocity
        self.graph_angle = np.array(kwargs['des_angle'], dtype=float) # Shape rotation (Euler angles)
        self.angle_vel = kwargs['angle_vel']                   # Angular velocity
        self.destination = np.array(kwargs['destination'], dtype=float)  # Destination point

        # UAV statistical states
        self.container = np.zeros(self.num, dtype=int)         # Grid points owned by each UAV
        self.avg_contain = np.zeros(self.num, dtype=int)       # Neighbor average container size
        self.avg_des = np.ones(self.num, dtype=float)          # Desired neighbor distance
        self.avg_dist = np.ones(self.num, dtype=float)         # Actual average distance
        self.nei_len = np.ones(self.num, dtype=float)          # Neighbor count
        self.pos_color = np.zeros(self.num, dtype=float)       # Whether each UAV is in the shape

        # Time and logging
        self.t1, self.t2 = 0, 0                                 # Timestamps
        self.total_move = 0                                     # Total movement over time
        self.record_rate = [[0, 0, 1]]                          # Rate recording
        self.current_time = 0                                   # Current simulation time

    def get_graph_pos_angle_vel(self):
        """
        Calculate the current graph velocity vector based on destination.
        This influences UAV movement if target shape is moving.
        """
        toward = self.destination - self.graph_center
        if np.linalg.norm(toward) > 0:
            self.graph_vel = toward / np.linalg.norm(toward) * 0  # Movement disabled (set to 0)
        elif np.linalg.norm(toward) > 0:
            self.graph_vel = toward
        else:
            self.graph_vel = np.zeros(3)
        return self.kd_graph, self.graph_vel

    def GetDynFormation(self):
        """
        Update the dynamic formation by rotating the shape and translating it to the graph center.
        Builds a new KDTree after transformation.
        """
        angles = np.radians(self.graph_angle)

        # Rotation matrices around x, y, z axes
        rotation_x = np.array([
            [1, 0, 0],
            [0, np.cos(angles[0]), -np.sin(angles[0])],
            [0, np.sin(angles[0]), np.cos(angles[0])]
        ])
        rotation_y = np.array([
            [np.cos(angles[1]), 0, np.sin(angles[1])],
            [0, 1, 0],
            [-np.sin(angles[1]), 0, np.cos(angles[1])]
        ])
        rotation_z = np.array([
            [np.cos(angles[2]), -np.sin(angles[2]), 0],
            [np.sin(angles[2]),  np.cos(angles[2]), 0],
            [0, 0, 1]
        ])

        # Apply combined rotation and translation
        rotated_points = np.dot(rotation_x, np.dot(rotation_y, np.dot(rotation_z, self.init_graph.T))).T
        self.graph = rotated_points + self.graph_center
        self.kd_graph = KDTree(self.graph)

    def get_neighbor(self, num):
        """
        Return the list of UAVs within sensing radius of UAV[num].
        Includes positional, velocity, and behavioral information.
        """
        target_point = self.pos[num]
        pos = np.array(self.pos)
        distance_to_target = np.linalg.norm(pos - target_point, axis=1)
        mask = (distance_to_target < self.r_sense) & (np.arange(self.num) != num)

        # Filter UAVs using the mask
        within_distance_points = self.pos[mask]
        within_distance_vel = self.vel[mask]
        container = self.container[mask]
        avg_contain = self.avg_contain[mask]
        avg_des = self.avg_des[mask]
        avg_dist = self.avg_dist[mask]
        nei_len = self.nei_len[mask]
        pos_color = self.pos_color[mask]

        return (
            np.array(within_distance_points),
            np.array(within_distance_vel),
            np.array(container),
            np.array(avg_contain),
            np.array(avg_des),
            np.array(avg_dist),
            np.array(nei_len),
            np.array(pos_color)
        )

    def input_container(self, index, num):
        """
        Store the number of shape points assigned to UAV[index].
        """
        self.container[index] = num


