# CUDA Batch Flood Fill

## Overview

The CUDA implementation accelerates Flood Fill by processing many mazes in parallel.

Instead of accelerating a single 16 × 16 maze, the CUDA version processes a batch of mazes.

Default batch size:

```text
10000 mazes
```

## CUDA Mapping

```text
blockIdx.x  = maze_id
threadIdx.x = cell_id
```

Each maze has:

```text
16 × 16 = 256 cells
```

Therefore:

```text
1 CUDA block = 1 maze
256 CUDA threads = 256 cells
```

## Memory Layout

Input:

```text
walls_batch[num_mazes * 256]
```

Output:

```text
distance_batch[num_mazes * 256]
```

Global index:

```text
global_idx = maze_id * 256 + cell_id;
```

## Input

The CUDA program reads:

```text
data/mazes_10000_seed0.bin
```

## Output

The CUDA program writes:

```text
results/gpu_distances_10000.bin
results/gpu_flood_fill_summary_shared.txt
```

## Build

From project root:

```bash
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

## Run

From build/:

```bash
.\cuda\Release\flood_fill_cuda.exe 10000 ..\data\mazes_10000_seed0.bin ..\results\gpu_distances_10000.bin ..\results\gpu_flood_fill_summary_shared.txt
```

Arguments:

```text
argv[1] = number of mazes
argv[2] = maze binary file
argv[3] = GPU distance output file
argv[4] = GPU summary output file
```

## Algorithm

The CPU version uses queue-based BFS.

The CUDA version uses iterative relaxation:

```text
distance[current] = min(distance[current], distance[neighbor] + 1)
```

Process:

```text
1. Load one maze into shared memory.
2. Initialize center goal cells to 0.
3. Initialize other cells to INF.
4. Each thread checks its four legal neighbors.
5. If a neighbor has a smaller distance, update current cell.
6. Repeat until no cell changes.
```

## Why Iterative Relaxation?

Queue-based BFS is not ideal for GPU because it involves:

```text
dynamic queue operations
irregular memory access
branch-heavy control flow
```

Iterative relaxation is easier to parallelize because every cell can update itself simultaneously.

## Summary File

The GPU summary records:

```text
maze_file
num_mazes
kernel_time_ms
kernel_time_sec
avg_time_per_maze_ms
distance_output_file
walls_bytes
distance_bytes
cuda_mapping
```

## Correctness

After running GPU benchmark, compare with CPU:

```bash
python -m experiments.cpu_gpu_comparison
```

Expected result:

```text
All equal: True
```
