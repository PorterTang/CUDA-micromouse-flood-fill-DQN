# Shared Maze Dataset

## Purpose

The project uses a shared maze dataset to ensure that CPU Flood Fill, GPU Flood Fill, and DQN evaluation are performed on the same maze set.

This avoids unfair comparison caused by different random mazes.

## Dataset File

Default dataset:

```text
data/mazes_10000_seed0.bin
```


Meta data:

```text
data/mazes_10000_seed0_meta.txt
```

## Format

The binary file stores maze wall data.

```text
shape = [10000, 256]
dtype = uint8
```

Each cell has 256 cells.

Each cell stroes 4-bit wall information:

```text
bit 0: North wall
bit 1: East wall
bit 2: South wall
bit 3: West wall
```

## Generate Dataset

Run:

```bash
python -m experiments.generate_shared_mazes
```

Expected output:

```text
data/mazes_10000_seed0.bin
data/mazes_10000_seed0_meta.txt
data/maze_png/maze_00000.png
data/maze_png/maze_00001.png
...
```

## Why Binary Format?

Binary format is used because:

It is compact.
It can be loaded quickly.
It can be directly passed to CUDA.
It avoids inconsistent maze generation between CPU and GPU runs.
File Size

For 10,000 mazes:

```text
10000 × 256 × 1 byte = 2,560,000 bytes
```

The expected file size is:

```text
2.56 MB
```

## Dataset Verification

The dataset is checked using:

```text
check_maze_file("data/mazes_10000_seed0.bin", 10000)
```

The check verifies:

```text
1. File exists
2. File size is correct
3. Number of maze cells is correct
```
