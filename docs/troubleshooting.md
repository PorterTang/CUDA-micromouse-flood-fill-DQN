# Troubleshooting

## 1. ModuleNotFoundError: No module named 'simulator'

### Cause

The script was executed directly by file path instead of module mode.

Wrong:

```cmd
python rl/training_dqn.py
```

Correct:

```bash
python -m rl.training_dqn
```

## Fix

Run from project root:

```bash
cd C:\Users\27615\Desktop\graduate_school\RL\final_proj
python -m rl.training_dqn
```

Make sure these files exist:

```text
simulator/__init__.py
rl/__init__.py
experiments/__init__.py
```

## 2. GPU distance file not found

Error:

```text
FileNotFoundError: results/gpu_distances_10000.bin
```

### Cause

GPU benchmark has not been executed, or output path is incorrect.

### Fix

From build/:

```bash
.\cuda\Release\flood_fill_cuda.exe 10000 ..\data\mazes_10000_seed0.bin ..\results\gpu_distances_10000.bin ..\results\gpu_flood_fill_summary_shared.txt
```

Then from project root:

```bash
python -m experiments.cpu_gpu_comparison
```

## 3. CPU/GPU results are not equal

### Possible Causes

```text
1. CPU and GPU used different maze files.
2. GPU output file is old.
3. CUDA kernel was not rebuilt after modification.
4. Maze wall encoding is inconsistent.
```

### Fix

Run complete workflow again:

```bash
python -m experiments.generate_shared_mazes
python -m experiments.benchmark_cpu_batch

cd build
cmake --build . --config Release
.\cuda\Release\flood_fill_cuda.exe 10000 ..\data\mazes_10000_seed0.bin ..\results\gpu_distances_10000.bin ..\results\gpu_flood_fill_summary_shared.txt

cd ..
python -m experiments.cpu_gpu_comparison
```

## 4. DQN model file not found

Error:

```text
FileNotFoundError: results/dqn_micromouse_shared_mazes.pth
```

### Cause

DQN training has not been run.

Fix

```bash
python -m rl.training_dqn
```

Then evaluate:

```bash
python -m experiments.evaluation_dqn
```

## 5. agent.load("model_path") error

Error:

```text
FileNotFoundError: No such file or directory: 'model_path'
```

### Cause

The code uses a string instead of a variable.

Wrong:

```text
agent.load("model_path")
```

Correct:

```text
agent.load(model_path)
```

## 6. draw_maze() got unexpected keyword argument 'show'

### Cause

visualization.py is outdated.

### Fix

Update draw_maze() to include:

```text
def draw_maze(..., show=True, ...):
```

Or remove:

```text
show=False
```

from the function call.

## 7. GIF export fails

### Cause

Missing dependencies.

### Fix

Install:

```bash
python -m pip install imageio pillow
```

## 8. CUDA executable not found

### Fix

Search executable:

```bash
dir /s flood_fill_cuda.exe
```

Then run the executable from the actual path.

Example:

```bash
.\cuda\Release\flood_fill_cuda.exe 10000 ..\data\mazes_10000_seed0.bin ..\results\gpu_distances_10000.bin ..\results\gpu_flood_fill_summary_shared.txt
```

## 9. PyTorch CUDA unavailable

Check:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

If output is:

```text
False
```

Possible causes:

```text
1. PyTorch CPU version installed.
2. NVIDIA driver issue.
3. Wrong Conda environment.
```

Check GPU driver:

```bash
nvidia-smi
```

## 10. Reward stuck during DQN training

Possible causes:

```text
1. Maze task too difficult.
2. Epsilon decays too fast.
3. Reward shaping is too sparse.
4. Agent keeps turning or hitting walls.
```

Suggested fixes:

```text
1. Train on fixed maze first.
2. Increase episodes.
3. Slow down epsilon decay.
4. Add success rate logging.
5. Tune reward design.
```
