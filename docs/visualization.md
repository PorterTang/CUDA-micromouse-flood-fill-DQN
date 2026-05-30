# Visualization

## Overview

This project supports several visualization outputs:

```text
Maze PNG
Flood Fill distance PNG
DQN path PNG
DQN path GIF
CPU Flood Fill path PNG/GIF
GPU Flood Fill path PNG/GIF
```

The visualization functions are implemented in:

```text
simulator/visualization.py
```

## Goal Area

The Micromouse goal area is the center four cells:

```text
(7, 7), (7, 8), (8, 7), (8, 8)
```

In generated PNG/GIF outputs, the goal area is highlighted with a red background.

## Export Maze PNG

Run:

```bash
python -m experiments.export_maze_png
```

Outputs:

```text
data/maze_png/maze_00000.png
data/maze_png/maze_00001.png
...
```

## Export Flood Fill Distance PNG

Run:

```bash
python -m experiments.export_distance_png
```

Outputs:

```text
results/distance_png/maze_00000_distance.png
results/distance_png/maze_00001_distance.png
...
```

These images show:

```text
maze walls
goal area
distance value in each cell
```

## DQN Path Visualization

DQN evaluation generates:

```text
results/dqn_eval_path_shared_maze0.png
results/dqn_eval_path_shared_maze0.gif
```

These files show the actual path taken by the trained DQN agent.

## CPU/GPU Flood Fill Path Visualization

CPU/GPU distance outputs contain distance maps, not real trajectories.

To visualize a path, the project reconstructs a path by:

```text
1. Start from cell 0.
2. Check legal neighbors.
3. Move to the neighbor with smaller distance.
4. Repeat until reaching the goal.
```

Run:

```bash
python -m experiments.export_cpu_gpu_path_visuals
```

Outputs:

```text
results/cpu_flood_fill_path_shared_maze0.png
results/cpu_flood_fill_path_shared_maze0.gif
results/cpu_flood_fill_path_shared_maze0_distance.png

results/gpu_flood_fill_path_shared_maze0.png
results/gpu_flood_fill_path_shared_maze0.gif
results/gpu_flood_fill_path_shared_maze0_distance.png
```

## Difference Between DQN Path and Flood Fill Path

```text
DQN path:
    Actual agent trajectory.

CPU/GPU Flood Fill path:
    Reconstructed shortest path from the distance map.
```

Therefore, DQN path may include:

```text
turning
exploration
revisiting
mistakes
longer path
```

Flood Fill path usually represents:

```text
shortest route to goal
```

## GIF Requirements

Install:

```bash
python -m pip install imageio pillow
```
