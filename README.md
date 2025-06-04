# 🧠 Bubble-Raft Inspired Shape-Assembly in Flying Robot Swarm  
**Simulation Codebase (Core-Only Version)**  
*For submission to Nature Machine Intelligence*

---

## 📦 Overview

This repository provides a simplified yet functional simulation of our **bubble-raft inspired swarm assembly algorithm**. It models 2D/3D shape formation and uniform spatial coverage in a drone swarm, using only core components independent of complex simulation environments (e.g., AirSim or Crazyflie).

> ✅ This version is intended for algorithmic reproduction only. Code for hardware deployment and AirSim integration is omitted due to dependency complexity.

---

## 🗂️ Project Structure

```
.
├── main.py                    # Main script to run formation simulation
├── uav_control.py            # Core UAV behavior and decision-making logic
├── swarm_transmission.py     # Simulated AirSim environment & neighbor query
├── matlab_test.py            # Evaluation metrics (coverage, convergence, velocity, etc.)
├── utils.py                  # File I/O for results logging
├── settings.py               # UAV position generation (random, square grid, etc.)
├── models/                   # Target formation data (.mat, .json, .jpg, .stl)
├── data/                     # Output logs and recorded simulation data
├── compare/                  # (Optional) Comparison algorithm implementations
├── minecraft demo/           # 3D voxel visualization interface (optional)
└── sim.html                  # (Unused placeholder)
```

---

## 🚀 Getting Started

### Dependencies

```bash
pip install numpy scipy ipywidgets
```

> Note: AirSim import remains in `utils.py` for compatibility but is not used in this version.

---

## 🧩 How It Works

- **main.py** invokes `drone_swarm_formation(...)`, which initializes a swarm of agents placed in a target area.
- Each UAV independently executes:
  - **Entering**: moving toward the shape region.
  - **Exploration**: balancing uniform coverage of shape points.
  - **Interaction**: maintaining distance from neighbors.
- The formation shape is loaded from `.json`, `.mat`, or `.stl` in `/models`.
- Metrics such as entering rate, convergence coverage, and velocity variance are computed in `matlab_test.py`.

---

## 📈 Evaluation Metrics

Produced via `matlab_test.all_test(...)`:
- **Entering Rate**: Fraction of drones inside shape boundary.
- **Convergence**: Percentage of target points covered.
- **Uniformity**: Spatial variance of pairwise distances.
- **Movement Cost**: Cumulative drone displacement.
- **Velocity Control**: Per-drone velocity stats.

All outputs are saved in `./data/run_data/`, including:

```
rate_km_*.json   # Coverage metrics
pos_km_*.json    # UAV positions over time
comm_km_*.json   # Communication behavior
time_km_*.json   # Time profiling
```

---

## 🧪 Example Usage

```bash
python main.py
```

To switch the target shape, modify the following in `main.py`:

```python
graph = read_gray_mtr('./models')  # For 2D grayscale image
# graph = read_stl('flower.stl')   # For 3D target shape
drone_swarm_formation(graph, 1000)
```

---

## 📌 Notes

- This simulation is resolution- and scale-sensitive. Shape point spacing and UAV sensing radius (`r_sense`) must be compatible.
- For fast testing, `swarm_size` and `rounds` can be reduced in `main.py`.

---

## 📃 Citation

If you use this code, please cite our paper (preprint or DOI link will be provided after acceptance).
