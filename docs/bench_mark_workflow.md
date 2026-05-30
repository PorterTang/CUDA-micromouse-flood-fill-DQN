# Benchmark Workflow

## Purpose

This document describes the complete benchmark workflow for the project.

The goal is to compare:

```text
CPU Flood Fill
CUDA Batch Flood Fill
DQN maze exploration
```

using the same shared maze dataset.

## Step 1: Generate Shared Mazes

```bash
python -m experiments.generate_shared_mazes
```

Output:

```text
data/mazes_10000_seed0.bin
```

## Step 2: Run CPU Flood Fill Benchmark

```bash
python -m experiments.benchmark_cpu_batch
```

Output:

```text
results/cpu_distances_10000.bin
results/cpu_flood_fill_summary_shared.txt
```

## Step 3: Build CUDA

```bash
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

## Step 4: Run GPU Flood Fill Benchmark

From build/:

```bash
.\cuda\Release\flood_fill_cuda.exe 10000 ..\data\mazes_10000_seed0.bin ..\results\gpu_distances_10000.bin ..\results\gpu_flood_fill_summary_shared.txt
```

Output:

```text
results/gpu_distances_10000.bin
results/gpu_flood_fill_summary_shared.txt
```

## Step 5: Compare CPU and GPU Results

From project root:

```bash
python -m experiments.cpu_gpu_comparison
```

Output:

```text
results/cpu_gpu_comparison_summary_shared.txt
```

Expected result:

```text
All equal: True
```

## Step 6: Export CPU/GPU Path Visuals

```bash
python -m experiments.export_cpu_gpu_path_visuals
```

Output:

```text
results/cpu_flood_fill_path_shared_maze0.png
results/cpu_flood_fill_path_shared_maze0.gif
results/gpu_flood_fill_path_shared_maze0.png
results/gpu_flood_fill_path_shared_maze0.gif
```

## Step 7: Train DQN

```bash
python -m rl.training_dqn
```

Output:

```text
results/dqn_micromouse_shared_mazes.pth
results/dqn_reward_curve_shared.png
```

## Step 8: Evaluate DQN

```bash
python -m experiments.evaluation_dqn
```

Output:

```text
results/dqn_eval_summary_shared.txt
results/dqn_eval_path_shared_maze0.png
results/dqn_eval_path_shared_maze0.gif
```

## Complete Command Sequence

```bash
cd C:\Users\27615\Desktop\graduate_school\RL\final_proj

python -m experiments.generate_shared_mazes

python -m experiments.benchmark_cpu_batch

mkdir build
cd build
cmake ..
cmake --build . --config Release

.\cuda\Release\flood_fill_cuda.exe 10000 ..\data\mazes_10000_seed0.bin ..\results\gpu_distances_10000.bin ..\results\gpu_flood_fill_summary_shared.txt

cd ..

python -m experiments.cpu_gpu_comparison

python -m experiments.export_cpu_gpu_path_visuals

python -m rl.training_dqn

python -m experiments.evaluation_dqn
```

## Expected Result Files

```text
results/
├── cpu_distances_10000.bin
├── gpu_distances_10000.bin
├── cpu_flood_fill_summary_shared.txt
├── gpu_flood_fill_summary_shared.txt
├── cpu_gpu_comparison_summary_shared.txt
├── dqn_micromouse_shared_mazes.pth
├── dqn_reward_curve_shared.png
├── dqn_eval_summary_shared.txt
├── dqn_eval_path_shared_maze0.png
└── dqn_eval_path_shared_maze0.gif
```