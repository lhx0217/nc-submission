import threading
import time
import numpy as np

import uav_control
import matlab_test
import utils
import settings
import swarm_transmission as airsim
from graph_base import read_gray_mtr  # or read_stl for 3D

# ==========================
# Global Parameters
# ==========================

SENSE_RADIUS = 10
SWARM_SIZE = 200
DESTINATION = [0, 0, 0]
GRAPH_VELOCITY = np.array([0.0, 0.0, 0.0])
ANGLE_VELOCITY = np.array([0.0, 0.0, 0.0])

# ==========================
# Initialize Simulation State
# ==========================

# Initial random UAV positions
initial_positions = settings.gen_settings(np.array([0, 0]), SWARM_SIZE)

# Placeholder for formation graph
formation_graph = []
graph_color = [1] * len(formation_graph)

# Construct initial state dictionary
init_state = {
    "r_sense": SENSE_RADIUS,
    "pos": initial_positions,
    "graph": formation_graph,
    "des_pos": DESTINATION,
    "des_angle": [0, 0, 0],
    "graph_color": graph_color,
    "graph_vel": GRAPH_VELOCITY,
    "angle_vel": ANGLE_VELOCITY,
    "destination": DESTINATION,
    "obstacles": [],
    "param": None
}

# Initialize simulation engine (simplified AirSim)
sim = airsim.AirSim(**init_state)

# ==========================
# Data Recording Lists
# ==========================

record_rate = []
record_pos = []
record_comm = []
record_time = []

stop_flag = False
destination_arrival_status = False


# ==========================
# Main Execution Functions
# ==========================

def run_uavs(uav_list, sim, init_state, total_rounds):
    """
    Run the main control loop for all UAVs in the swarm.
    """
    global stop_flag
    start_time = time.time()
    print(f'Executing {total_rounds} rounds...')

    for round_id in range(total_rounds):
        current_time = time.time()

        sim.GetDynFormation()

        for uav in uav_list:
            uav.run(round_id)

        sim.current_time = init_state['t'] * round_id

        # Record data for evaluation
        if round_id >= 0:
            rec, comm, center = matlab_test.all_test(sim, init_state, uav_list)
            record_rate.append(rec)
            record_pos.append(sim.pos.tolist())
            record_comm.append(comm)
            record_time.append(current_time - start_time)

        # Save every 5 steps asynchronously
        if (round_id + 1) % 5 == 0:
            thread = threading.Thread(
                target=utils.file_write,
                args=(init_state['kmeans'], record_rate, record_pos, record_comm, record_time)
            )
            thread.start()


def drone_swarm_formation(graph, num_rounds):
    """
    Initialize the swarm and start shape assembly process.
    """
    global sim, stop_flag

    stop_flag = False
    if isinstance(graph, dict):
        graph = graph['graph']

    # Update formation target
    sim.init_graph = graph

    # Set updated parameters
    init_state = {
        "r_sense": SENSE_RADIUS,
        "r_avoid": 1.0,
        "t": 0.05,
        "pos": sim.pos,
        "graph": graph,
        "des_pos": sim.graph_center,
        "des_angle": sim.graph_angle,
        "graph_color": [1] * len(graph),
        "kmeans": 1,
        "graph_vel": [0, 0, 0],
        "angle_vel": [0, 0, 0],
        "destination": DESTINATION,
        "obstacles": [],
        "param": None
    }

    # Create UAV instances
    uav_list = [uav_control.UAV(i, sim, **init_state) for i in range(SWARM_SIZE)]

    # Start the control loop
    run_uavs(uav_list, sim, init_state, num_rounds)


# ==========================
# Standalone Execution
# ==========================

if __name__ == "__main__":
    # Example: Load a target formation from 2D grayscale matrix
    graph = read_gray_mtr('./models')

    # Alternative: Uncomment to load a 3D STL model
    # graph = read_stl('./models/line_sphere.stl')

    # Start the swarm formation process for 1000 time steps
    drone_swarm_formation(graph, 1000)
