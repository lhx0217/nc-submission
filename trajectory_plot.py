import json
import numpy as np
import matplotlib.pyplot as plt
import imageio
import os

# ---------- 工具函数 ----------
def loadjson(path):
    with open(path, 'r') as f:
        return json.load(f)

# ---------- 初始化 ----------
data_files = ['pos_km_dragon.json', 'pos_ms_dragon.json', 'heal_pos_dragon.json', 'sci_pos_dragon.json']
titles = ['Our Proposed', 'Mean-shift', 'Image Moment', 'Graph Similarity']
flip_ys = [True, True, True, True]
swap_xys = [False, False, False, True]
num_plots = 4

# 读取graph数据
graph = np.array(loadjson('./models/dragon.json'))
graph[:, 1] = -graph[:, 1]

# 读取所有轨迹数据
pos_datas = []
max_steps = 0
max_agents = []

for file in data_files:
    data = np.array(loadjson(os.path.join('./data/run_data', file)))
    pos_datas.append(data)
    max_steps = max(max_steps, len(data))
    max_agents.append(data.shape[1])

# ---------- 绘图准备 ----------
start_color = [0.35, 0.35, 0.35]
end_color = [0, 0.4470, 0.7410]
traj_alpha = 0.5
current_marker_size = 50
gif_path = './data/run_data/dragon_4plots_2D.gif'

# ---------- 初始化图像 ----------
fig, axes = plt.subplots(1, num_plots, figsize=(16, 4), dpi=100)
plots = []

for k in range(num_plots):
    ax = axes[k]
    ax.set_title(titles[k], fontsize=18)
    ax.set_xlim([-35, 35])
    ax.set_ylim([-35, 35])
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_xlabel('X (m)', fontsize=14)
    ax.set_ylabel('Y (m)', fontsize=14)
    ax.scatter(graph[:, 0], graph[:, 1], s=30, c='g', alpha=0.08)

    num_agents = max_agents[k]
    traj_data = [np.full((max_steps, 2), np.nan) for _ in range(num_agents)]

    lines = []
    colors = plt.cm.get_cmap('tab10', num_agents)

    for i in range(num_agents):
        line, = ax.plot([], [], lw=1, color=colors(i), alpha=0.15)
        lines.append(line)

    h_start = ax.scatter([], [], s=10, c=[start_color]*num_agents)
    h_robot = ax.scatter([], [], s=current_marker_size, c=[end_color])

    pos_data = pos_datas[k]
    start_pts = []
    for i in range(num_agents):
        pt = np.array(pos_data[0][i][:2])
        if swap_xys[k]:
            pt[0], pt[1] = pt[1], pt[0]
        if flip_ys[k]:
            pt[1] = -pt[1]
        start_pts.append(pt)
    start_pts = np.array(start_pts)
    h_start.set_offsets(start_pts)

    plots.append({
        'traj_data': traj_data,
        'lines': lines,
        'h_robot': h_robot,
        'pos_data': pos_data,
        'flip_y': flip_ys[k],
        'swap_xy': swap_xys[k],
    })

# ---------- 生成动画帧 ----------
frames = []

for t in range(400):
    print(f'Step {t+1}')
    for k in range(num_plots):
        info = plots[k]
        num_agents = len(info['lines'])
        pos_now = np.zeros((num_agents, 2))

        for i in range(num_agents):
            if t < len(info['pos_data']):
                pt = np.array(info['pos_data'][t][i][:2])
                if info['swap_xy']:
                    pt[0], pt[1] = pt[1], pt[0]
                if info['flip_y']:
                    pt[1] = -pt[1]
            else:
                pt = [np.nan, np.nan]
            info['traj_data'][i][t] = pt
            traj = info['traj_data'][i][:t+1]
            info['lines'][i].set_data(traj[:, 0], traj[:, 1])
            pos_now[i] = pt

        info['h_robot'].set_offsets(pos_now)

    plt.tight_layout()
    fig.canvas.draw()
    frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    frames.append(frame)

# ---------- 写入GIF ----------
if os.path.exists(gif_path):
    os.remove(gif_path)
imageio.mimsave(gif_path, frames, duration=0.05)
print(f'GIF saved to {gif_path}')
