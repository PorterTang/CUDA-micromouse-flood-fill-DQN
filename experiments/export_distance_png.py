"""
export_distance_png.py

Export maze PNG images with CPU Flood Fill distance values.

This script reads:

    data/mazes_10000_seed0.bin

computes CPU Flood Fill distance maps, and exports:

    results/distance_png/maze_00000_distance.png
    results/distance_png/maze_00001_distance.png
    ...

Run:

    python -m experiments.export_distance_png
"""

import os

from simulator.maze_io import load_maze_objects, check_maze_file
from simulator.flood_fill import flood_fill
from simulator.visualization import save_maze_png


def export_distance_png(
    maze_file="data/mazes_10000_seed0.bin",
    num_mazes=10000,
    num_png=5,
    output_dir="results/distance_png",
):
    """
    Export maze images with Flood Fill distance values.

    Args:
        maze_file:
            Shared maze binary file.
        num_mazes:
            Number of mazes stored in the binary file.
        num_png:
            Number of distance-map PNG images to export.
        output_dir:
            Output directory for distance PNG images.
    """

    print("Export Flood Fill Distance PNG")
    print("------------------------------")
    print(f"Maze file: {maze_file}")
    print(f"Number of mazes in file: {num_mazes}")
    print(f"Number of PNGs to export: {num_png}")
    print(f"Output directory: {output_dir}")
    print()

    check_maze_file(maze_file, num_mazes)

    print("Loading maze objects...")
    mazes = load_maze_objects(maze_file, num_mazes)
    print("Maze loading finished.")
    print()

    os.makedirs(output_dir, exist_ok=True)

    png_count = min(num_png, num_mazes)

    for i in range(png_count):
        maze = mazes[i]

        print(f"Computing Flood Fill distance for maze {i:05d}...")

        distance = flood_fill(maze)

        save_path = os.path.join(
            output_dir,
            f"maze_{i:05d}_distance.png",
        )

        save_maze_png(
            maze=maze,
            save_path=save_path,
            distance=distance,
            title=f"Maze {i:05d} Flood Fill Distance",
            show_index=False,
        )

        print(f"Saved: {save_path}")

    print()
    print("Distance PNG export finished.")


def main():
    export_distance_png(
        maze_file="data/mazes_10000_seed0.bin",
        num_mazes=10000,
        num_png=20,
        output_dir="results/distance_png",
    )


if __name__ == "__main__":
    main()