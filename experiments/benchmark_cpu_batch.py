"""
benchmark_cpu_batch.py

CPU Flood Fill benchmark using shared maze dataset.

Run:
    python -m experiments.benchmark_cpu_batch
"""

import os
import time
import numpy as np

from simulator.maze_io import load_maze_objects, check_maze_file
from simulator.flood_fill import flood_fill


def benchmark_cpu_batch(
    num_mazes=10000,
    maze_file="data/mazes_10000_seed0.bin",
    distance_output_file="results/cpu_distances_10000.bin",
    summary_output_file="results/cpu_flood_fill_summary_shared.txt",
):
    """
    Benchmark CPU Flood Fill on the shared maze dataset.

    This function:
    1. Loads the shared maze dataset.
    2. Runs CPU Flood Fill on all mazes.
    3. Saves all distance maps to binary file.
    4. Saves benchmark summary to txt file.
    """

    os.makedirs("results", exist_ok=True)

    print("CPU Batch Flood Fill Benchmark")
    print("------------------------------")
    print(f"Maze file: {maze_file}")
    print(f"Number of mazes: {num_mazes}")
    print(f"Distance output file: {distance_output_file}")
    print(f"Summary output file: {summary_output_file}")
    print()

    check_maze_file(maze_file, num_mazes)

    print()
    print("Loading maze objects...")
    load_start = time.perf_counter()

    mazes = load_maze_objects(maze_file, num_mazes)

    load_end = time.perf_counter()
    load_time = load_end - load_start

    print(f"Load time: {load_time:.6f} sec")

    all_distances = np.zeros((num_mazes, 256), dtype=np.int32)

    print()
    print("Running CPU Flood Fill...")

    compute_start = time.perf_counter()

    for idx_maze, maze in enumerate(mazes):
        all_distances[idx_maze, :] = flood_fill(maze)

    compute_end = time.perf_counter()
    compute_time = compute_end - compute_start

    avg_time_per_maze_ms = compute_time / num_mazes * 1000.0

    all_distances.tofile(distance_output_file)

    print()
    print("CPU Benchmark Result")
    print("--------------------")
    print(f"Load time: {load_time:.6f} sec")
    print(f"CPU compute time: {compute_time:.6f} sec")
    print(f"Average per maze: {avg_time_per_maze_ms:.6f} ms")
    print(f"Distance output saved to: {distance_output_file}")

    print()
    print("First maze CPU distance map:")

    first_distance = all_distances[0]

    for row in range(16):
        row_values = []

        for col in range(16):
            idx = row * 16 + col
            row_values.append(f"{first_distance[idx]:3d}")

        print(" ".join(row_values))

    with open(summary_output_file, "w", encoding="utf-8") as f:
        f.write("CPU Flood Fill Benchmark Result\n")
        f.write("-------------------------------\n")
        f.write(f"maze_file={maze_file}\n")
        f.write(f"num_mazes={num_mazes}\n")
        f.write(f"maze_size=16\n")
        f.write(f"num_cells=256\n")
        f.write(f"load_time_sec={load_time:.6f}\n")
        f.write(f"compute_time_sec={compute_time:.6f}\n")
        f.write(f"avg_time_per_maze_ms={avg_time_per_maze_ms:.6f}\n")
        f.write(f"distance_output_file={distance_output_file}\n")
        f.write(f"distance_output_shape=({num_mazes}, 256)\n")
        f.write("distance_output_dtype=int32\n")

    print(f"Summary saved to: {summary_output_file}")


def main():
    benchmark_cpu_batch(
        num_mazes=10000,
        maze_file="data/mazes_10000_seed0.bin",
        distance_output_file="results/cpu_distances_10000.bin",
        summary_output_file="results/cpu_flood_fill_summary_shared.txt",
    )


if __name__ == "__main__":
    main()