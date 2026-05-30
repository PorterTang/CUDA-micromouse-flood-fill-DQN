# CUDA-Accelerated Micromouse Maze Solver with Flood Fill and Reinforcement Learning-Based Exploration

This project implements a Micromouse maze-solving system that integrates:

1. **1D-array maze representation**
2. **CPU Flood Fill path planning**
3. **CUDA Batch Flood Fill acceleration**
4. **DQN-based reinforcement learning exploration**
5. **Shared maze dataset for fair CPU / GPU / RL benchmarking**
6. **Maze distance map, path PNG, and GIF visualization**

The main goal is to evaluate classical Flood Fill, GPU-accelerated batch computation, and reinforcement learning-based exploration on the same Micromouse maze dataset.

---

## 1. Project Overview

Micromouse is a classic autonomous robot problem where an agent must explore and solve a maze. In this project, each maze is represented as a 16 × 16 grid.

Each cell stores wall information using 4 bits:

| Bit | Direction |
|---:|---|
| bit 0 | North wall |
| bit 1 | East wall |
| bit 2 | South wall |
| bit 3 | West wall |

Instead of using a 2D array such as:

```text
maze[row][col]

this project uses a 1D array:
maze[idx]
```

where:

```text
idx = row * 16 + col
row = idx // 16
col = idx % 16
```

For a 16 × 16 maze:

```text
idx range = 0 ~ 255
```

This design is used consistently across CPU, GPU, and RL modules.

## 2. Key Features

### Maze Representation
● 16 × 16 Micromouse maze
● 256 cells per maze
● 1D array-based wall representation
● 4-bit wall encoding per cell

### CPU Flood Fill
● Standard BFS-based Flood Fill
● Computes distance from every cell to the center goal
● Used as the correctness baseline

### CUDA Batch Flood Fill
● One CUDA block processes one maze
● One CUDA thread processes one cell
● 256 threads per maze
● Uses shared memory and iterative relaxation
● Designed for large-scale batch processing

### DQN Reinforcement Learning
● Custom Micromouse environment
● DQN agent using PyTorch
● Uses the same shared maze dataset as CPU and GPU
● Outputs reward curve, evaluation summary, path PNG, and GIF

### Visualization
● Maze PNG export
● Flood Fill distance PNG export
● DQN evaluation path PNG/GIF
● CPU/GPU Flood Fill path PNG/GIF reconstructed from distance .bin files

## 3. Project Structure

```text
final_proj/
│
├── cuda/
│   ├── CMakeLists.txt
│   └── flood_fill_cuda.cu
│
├── data/
│   ├── mazes_10000_seed0.bin
│   ├── mazes_10000_seed0_meta.txt
│   └── maze_png/
│       ├── maze_00000.png
│       ├── maze_00001.png
│       └── ...
│
├── experiments/
│   ├── __init__.py
│   ├── generate_shared_mazes.py
│   ├── benchmark_cpu_batch.py
│   ├── cpu_flood_fill.py
│   ├── cpu_gpu_comparison.py
│   ├── evaluation_dqn.py
│   ├── export_maze_png.py
│   ├── export_distance_png.py
│   └── export_cpu_gpu_path_visuals.py
│
├── rl/
│   ├── __init__.py
│   ├── dqn_agent.py
│   ├── replay_buffer.py
│   └── training_dqn.py
│
├── simulator/
│   ├── __init__.py
│   ├── maze.py
│   ├── maze_generator.py
│   ├── maze_io.py
│   ├── flood_fill.py
│   ├── micromouse_env.py
│   └── visualization.py
│
├── results/
│   ├── cpu_distances_10000.bin
│   ├── gpu_distances_10000.bin
│   ├── cpu_flood_fill_summary_shared.txt
│   ├── gpu_flood_fill_summary_shared.txt
│   ├── cpu_gpu_comparison_summary_shared.txt
│   ├── dqn_micromouse_shared_mazes.pth
│   ├── dqn_reward_curve_shared.png
│   ├── dqn_eval_path_shared_maze0.png
│   ├── dqn_eval_path_shared_maze0.gif
│   ├── cpu_flood_fill_path_shared_maze0.png
│   ├── cpu_flood_fill_path_shared_maze0.gif
│   ├── gpu_flood_fill_path_shared_maze0.png
│   └── gpu_flood_fill_path_shared_maze0.gif
│
├── CMakeLists.txt
├── requirement.txt
└── README.md
```

## 4. Environment Setup

### Conda Environment

The project is designed to run in a Python environment with PyTorch GPU support.

Example environment name:

```text
pytorch_gpu
```

Activate the environment:

```bash
conda activate pytorch_gpu
```

Install dependencies:

```bash
python -m pip install -r requirement.txt
```

Check PyTorch CUDA:

```bash
python -c "import torch; print('torch version:', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

Expected output:

```text
cuda available: True
gpu: NVIDIA ...
```

## 5. Generate Shared Maze Dataset

CPU, GPU, and RL all use the same maze dataset.

Generate 10,000 DFS mazes:

```bash
cd <your_path>

python -m experiments.generate_shared_mazes
```

This will generate:

```text
data\mazes_10000_seed0.bin
data\mazes_10000_seed0_meta.txt
data\maze_png\maze_00000.png
data\maze_png\maze_00001.png
...
```

The binary file stores:

```text
shape = [10000, 256]
dtype = uint8
```

Each maze is represented as:

```text
walls[256]
```

## 6. CPU Flood Fill Benchmark

Run CPU benchmark on the shared maze dataset:

```bash
python -m experiments.benchmark_cpu_batch
```

Output:

```text
results\cpu_distances_10000.bin
results\cpu_flood_fill_summary_shared.txt
```

The CPU distance binary file stores:

```text
shape = [10000, 256]
dtype = int32
```

The summary file records:

```text
maze_file
num_mazes
load_time_sec
compute_time_sec
avg_time_per_maze_ms
distance_output_file
```

## 7. Build CUDA Program

Create and enter build directory:

```bash
cd <your_path>

mkdir build

cd build
```

Configure CMake:

```bash
cmake ..
```

Build Release version:

```bash
cmake --build . --config Release
```

The CUDA executable should be generated at:

```text
build\cuda\Release\flood_fill_cuda.exe
```

If the executable path is different, search it by:

```bash
dir /s flood_fill_cuda.exe
```

## 8. GPU Flood Fill Benchmark

Run CUDA benchmark using the same shared maze dataset:

```bash
cd <your_path>

.\cuda\Release\flood_fill_cuda.exe 10000 ..\data\mazes_10000_seed0.bin ..\results\gpu_distances_10000.bin
```

Arguments:

```text
argv[1] = number of mazes
argv[2] = shared maze binary file
argv[3] = GPU distance output file
argv[4] = GPU benchmark summary output file
```

Outputs:

```text
results\gpu_distances_10000.bin
results\gpu_flood_fill_summary_shared.txt
```

The GPU distance binary file stores:

```text
shape = [10000, 256]
dtype = int32
```

## 9. Compare CPU and GPU Results

After CPU and GPU benchmarks are complete, compare the results:

```bash
cd <your_path>

python -m experiments.cpu_gpu_comparison
```

Expected result:

```text
CPU/GPU Distance Comparison
---------------------------
All equal: True
CPU and GPU results are identical.
```

Output:

```text
results\cpu_gpu_comparison_summary_shared.txt
```

This confirms that the CUDA Flood Fill output matches the CPU baseline.

## 10. DQN Training

Train the DQN agent using the same shared maze dataset:

```bash
cd <your_path>

python -m rl.training_dqn
```

Output files:

```text
results\dqn_micromouse_shared_mazes.pth
results\dqn_rewards_shared.npy
results\dqn_losses_shared.npy
results\dqn_success_shared.npy
results\dqn_steps_shared.npy
results\dqn_reward_curve_shared.png
```

The training environment uses:

```text
data\mazes_10000_seed0.bin
```

so the RL agent is trained on the same maze set used by CPU and GPU benchmarks.

## 11. DQN Evaluation

Evaluate the trained DQN agent:

```bash
python -m experiments.evaluation_dqn
```

Output files:

```text
results\dqn_eval_path_shared_maze0.png
results\dqn_eval_path_shared_maze0.gif
results\dqn_eval_summary_shared.txt
results\dqn_eval_steps_shared.npy
results\dqn_eval_rewards_shared.npy
```

Evaluation metrics include:

```text
Success rate
Average steps
Average reward
Average inference/evaluation time per maze
```

## 12. Export Maze PNG Images

If the shared maze binary file already exists and you only want to export PNG images:

```bash
python -m experiments.export_maze_png
```

This generates:

```text
data\maze_png\maze_00000.png
data\maze_png\maze_00001.png
...
```

## 13. Export Flood Fill Distance PNG Images

To visualize Flood Fill distance maps:

```bash
python -m experiments.export_distance_png
```

This generates:

```text
results\distance_png\maze_00000_distance.png
results\distance_png\maze_00001_distance.png
...
```

## 14. Export CPU/GPU Flood Fill Path PNG and GIF

The CPU/GPU distance .bin files contain distance maps, not real exploration trajectories.

To visualize a path, this project reconstructs a shortest path by starting from cell 0 and repeatedly moving to the legal neighbor with a smaller Flood Fill distance.

Run:

```bash
python -m experiments.export_cpu_gpu_path_visuals
```

Inputs:

```text
data\mazes_10000_seed0.bin
results\cpu_distances_10000.bin
results\gpu_distances_10000.bin
```

Outputs:

```text
results\cpu_flood_fill_path_shared_maze0.png
results\cpu_flood_fill_path_shared_maze0.gif
results\cpu_flood_fill_path_shared_maze0_distance.png

results\gpu_flood_fill_path_shared_maze0.png
results\gpu_flood_fill_path_shared_maze0.gif
results\gpu_flood_fill_path_shared_maze0_distance.png

results\cpu_gpu_path_visual_summary_maze0.txt
```

Difference from DQN visualization:

```text
DQN path:
    Actual agent trajectory.

CPU/GPU Flood Fill path:
    Reconstructed shortest path from the distance map.
```

## 15. Complete Execution Flow

The recommended full execution flow is:

```bash
cd <your_path>

where python

python -c "import torch; print('torch version:', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"

python -m pip install -r requirement.txt

python -m experiments.generate_shared_mazes

python -m experiments.benchmark_cpu_batch

mkdir build

cd build

cmake ..

cmake --build . --config Release

.\cuda\Release\flood_fill_cuda.exe 10000 ..\data\mazes_10000_seed0.bin ..\results\gpu_distances_10000.bin

cd ..

python -m experiments.cpu_gpu_comparison

python -m rl.training_dqn

python -m experiments.evaluation_dqn
```

## 16. CUDA Mapping Design

The CUDA implementation uses this mapping:

```text
blockIdx.x  = maze_id
threadIdx.x = cell_id
```

Each maze has:

```text
16 x 16 = 256 cells
```

Therefore:

```text
1 CUDA block = 1 maze
256 CUDA threads = 256 cells
```

For batch memory layout:

```text
walls_batch[num_mazes * 256]
distance_batch[num_mazes * 256]
```

The global index is:

```text
global_idx = maze_id * 256 + cell_id;
```

## 17. CUDA Flood Fill Algorithm

The CPU version uses BFS-based Flood Fill.

The CUDA version uses parallel iterative relaxation:

```text
distance[current] = min(distance[current], distance[neighbor] + 1)
```

Process:

1. Initialize center goal cells as 0.
2. Initialize all other cells as INF.
3. Each thread checks its four neighbors.
4. If a neighbor distance + 1 is smaller, update current cell distance.
5. Repeat until no cell changes.

This approach avoids queue-based BFS, which is harder to parallelize efficiently on GPU.

## 18. Why Use Shared Maze Dataset?

To ensure fair comparison, all CPU, GPU, and RL experiments use the same maze dataset.

```text
data\mazes_10000_seed0.bin
```

This prevents inconsistent results caused by different random mazes.

The benchmark workflow is:

```text
Generate shared mazes
        ↓
CPU Flood Fill benchmark
        ↓
GPU Flood Fill benchmark
        ↓
CPU/GPU result comparison
        ↓
DQN training and evaluation
```

## 19. Expected Results
CPU/GPU Correctness

Expected CPU/GPU comparison:

```text
All equal: True
GPU Benchmark
```

The GPU benchmark should show:

```text
GPU kernel time
Average per maze
RL Evaluation
```

The RL evaluation should show:

```text
Success Rate
Average Steps
Average Reward
```

## 20. Notes

### Why 1D Array?

A 1D array is used because it is easier to transfer to CUDA and more suitable for batch processing.

Instead of:

```text
walls[row][col]
```

this project uses:

```text
walls[idx]
```

where:

```text
idx = row * 16 + col
```

### Why CUDA Batch Processing?

A single 16 × 16 maze is too small for GPU acceleration.

Therefore, CUDA is used for batch processing:

```text
10000 mazes
10000 CUDA blocks
256 threads per block
```

This makes GPU parallelism meaningful.

### Why Not Use Torch for Flood Fill?

Flood Fill contains queue logic, condition checks, and irregular memory access.

Therefore:

```text
Maze simulation / Flood Fill: NumPy / C++ / CUDA
DQN training: PyTorch CUDA
```

## 20. Suggested Git Ignore

Recommended .gitignore:

```gitignore
# Python cache
__pycache__/
*.pyc

# Build output
build/

# Large generated data
data/*.bin
results/*.bin
results/*.npy
results/*.pth

# Optional large image dumps
data/maze_png/*.png
results/distance_png/*.png

# Keep selected demo images if needed
!data/maze_png/maze_00000.png
!data/maze_png/maze_00001.png
!results/distance_png/maze_00000_distance.png
!results/dqn_reward_curve_shared.png
!results/dqn_eval_path_shared_maze0.png
```
