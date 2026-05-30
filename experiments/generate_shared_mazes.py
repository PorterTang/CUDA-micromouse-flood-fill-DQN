"""
generate_shared_mazes.py

Generate a shared maze dataset for CPU, GPU, and RL.

All modules will use the same binary maze file:

    data/mazes_10000_seed0.bin

Run:

    python -m experiments.generate_shared_mazes
"""

import os
import time

from simulator.maze_generator import generate_batch_mazes
from simulator.maze_io import save_maze_batch, check_maze_file


def generate_shared_mazes(num_mazes=10000, seed=0, output_path="data/mazes_10000_seed0.bin",):
    """
    Generate shared DFS maze dataset.

    Args:
        num_mazes:
            number of mazes
        seed:
            base random seed
        output_path:
            output binary file path
    """

    os.makedirs("data", exist_ok=True)

    print("Generating shared maze dataset...")
    print(f"Number of mazes: {num_mazes}")
    print(f"Seed: {seed}")
    print(f"Output path: {output_path}")

    start = time.perf_counter()

    mazes = generate_batch_mazes(num_mazes=num_mazes, seed=seed,)

    walls_batch = save_maze_batch(mazes, output_path)

    end = time.perf_counter()

    meta_path = output_path.replace(".bin", "_meta.txt")

    with open(meta_path, "w", encoding="utf-8") as f:
        f.write("Shared Micromouse Maze Dataset\n")
        f.write(f"num_mazes={num_mazes}\n")
        f.write("maze_size=16\n")
        f.write("num_cells=256\n")
        f.write("dtype=uint8\n")
        f.write(f"seed={seed}\n")
        f.write(f"binary_file={output_path}\n")
        f.write(f"shape={walls_batch.shape}\n")
        f.write(f"generation_time_sec={end - start:.6f}\n")

    check_maze_file(output_path, num_mazes)

    print()
    print("Generation finished.")
    print(f"Shape: {walls_batch.shape}")
    print(f"Generation time: {end - start:.6f} sec")
    print(f"Meta file: {meta_path}")


def main():
    generate_shared_mazes(
        num_mazes=10000,
        seed=0,
        output_path="data/mazes_10000_seed0.bin",
    )


if __name__ == "__main__":
    main()