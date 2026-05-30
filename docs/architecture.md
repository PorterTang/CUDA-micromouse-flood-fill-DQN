# Project Architecture

## Overview

This project implements a CUDA-accelerated Micromouse maze-solving system with CPU Flood Fill, GPU batch Flood Fill, and DQN-based reinforcement learning exploration.

The system is designed around a shared maze dataset so that CPU, GPU, and DQN experiments are performed on the same maze set.

## Main Components

```text
final_proj/
│
├── simulator/
│   ├── maze.py
│   ├── maze_generator.py
│   ├── maze_io.py
│   ├── flood_fill.py
│   ├── micromouse_env.py
│   └── visualization.py
│
├── experiments/
│   ├── generate_shared_mazes.py
│   ├── benchmark_cpu_batch.py
│   ├── cpu_gpu_comparison.py
│   ├── evaluation_dqn.py
│   ├── export_maze_png.py
│   ├── export_distance_png.py
│   └── export_cpu_gpu_path_visuals.py
│
├── rl/
│   ├── dqn_agent.py
│   ├── replay_buffer.py
│   └── training_dqn.py
│
├── cuda/
│   └── flood_fill_cuda.cu
│
├── data/
├── results/
└── docs/
```

## Data Flow

```text
Generate shared mazes
        ↓
data/mazes_10000_seed0.bin
        ↓
CPU Flood Fill benchmark
        ↓
results/cpu_distances_10000.bin
        ↓
GPU CUDA Flood Fill benchmark
        ↓
results/gpu_distances_10000.bin
        ↓
CPU/GPU comparison
        ↓
DQN training and evaluation
        ↓
PNG/GIF visualization
```

## Maze Representation

The maze is represented using a one-dimensional array.

For a 16 × 16 maze:

```text
num_cells = 16 × 16 = 256
```

The index conversion is:

```text
idx = row * 16 + col
row = idx // 16
col = idx % 16
```

Each cell stores wall information using 4 bits:

| Bit | Direction |
|------|------|
| bit0 | North |
| bit1 | East |
| bit2 | South |
| bit3 | West |

This format is used by CPU, GPU, and RL modules.