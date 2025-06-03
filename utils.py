import os

def file_write(kmeans, record_rate, record_pos, record_comm, record_time):
    """
    Save swarm simulation log data to JSON format files for analysis.

    Args:
        kmeans (bool): Whether the run used KMeans-assisted control.
        record_rate (list): List of coverage rates over time.
        record_pos (list): List of UAV position snapshots.
        record_comm (list): List of communication states.
        record_time (list): Time taken per round or checkpoint.
    """
    t = 'dragon'  # Dataset or shape name (fixed here, can be parameterized)

    if kmeans:
        base_path = './data/run_data/'
        prefix = 'km'
    else:
        base_path = '../../data/run_data/'
        prefix = 'ms'

    with open(os.path.join(base_path, f'rate_{prefix}_{t}.json'), 'w+') as f:
        f.write(str(record_rate))
    with open(os.path.join(base_path, f'pos_{prefix}_{t}.json'), 'w+') as f:
        f.write(str(record_pos))
    with open(os.path.join(base_path, f'comm_{prefix}_{t}.json'), 'w+') as f:
        f.write(str(record_comm))
    with open(os.path.join(base_path, f'time_{prefix}_{t}.json'), 'w+') as f:
        f.write(str(record_time))
