# CPU Flood Fill

## Overview

The CPU Flood Fill module computes the shortest distance from every cell to the Micromouse goal region.

The goal region is the center four cells:

```text
(7, 7), (7, 8), (8, 7), (8, 8)
```

For a 16 × 16 maze, the output is a distance map with 256 values.

## Input

The CPU benchmark reads the shared maze dataset:

```text
data/mazes_10000_seed0.bin
```

## Output

The CPU benchmark generates:

```text
results/cpu_distances_10000.bin
results/cpu_flood_fill_summary_shared.txt
```

Distance output format:

```text
shape = [10000, 256]
dtype = int32
```

## Run CPU Benchmark

```bash
python -m experiments.benchmark_cpu_batch
```

## Algorithm

CPU Flood Fill is based on breadth-first search.

Basic idea:

```text
1. Initialize goal cells with distance 0.
2. Push goal cells into queue.
3. Pop current cell from queue.
4. Visit all legal neighbors.
5. If neighbor distance can be improved, update it.
6. Continue until queue is empty.
```

## Distance Meaning

If:

```text
distance[idx] = 10
```

it means:

```text
The shortest path from cell idx to the goal takes 10 steps.
```

## Benchmark Summary

The summary file records:

```text
maze_file
num_mazes
maze_size
num_cells
load_time_sec
compute_time_sec
avg_time_per_maze_ms
distance_output_file
distance_output_shape
distance_output_dtype
```

## Role in This Project

The CPU Flood Fill result is used as the correctness baseline for the CUDA implementation.

The GPU output must match the CPU output exactly.