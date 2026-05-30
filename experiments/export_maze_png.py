"""
export_maze_png.py

Export selected maze images from the shared maze binary file.

This script reads:

    data/mazes_10000_seed0.bin

and exports selected mazes as PNG images:

    data/maze_png/maze_00000.png
    data/maze_png/maze_00001.png
    ...

Run:

    python -m experiments.export_maze_png
"""

import os

from simulator.maze_io import load_maze_objects, check_maze_file
from simulator.visualization import save_maze_png


def export_maze_png(
    maze_file="data/mazes_10000_seed0.bin",
    num_mazes=10000,
    num_png=20,
    png_dir="data/maze_png",
    show_index=False,
):
    """
    Export selected maze PNG images.

    Args:
        maze_file:
            Shared maze binary file.
        num_mazes:
            Number of mazes stored in the binary file.
        num_png:
            Number of maze PNG images to export.
        png_dir:
            Output directory for PNG images.
        show_index:
            If True, show cell idx inside each cell.
    """

    print("Export Maze PNG")
    print("---------------")
    print(f"Maze file: {maze_file}")
    print(f"Number of mazes in file: {num_mazes}")
    print(f"Number of PNGs to export: {num_png}")
    print(f"PNG directory: {png_dir}")
    print(f"Show cell index: {show_index}")
    print()

    check_maze_file(maze_file, num_mazes)

    print("Loading maze objects...")
    mazes = load_maze_objects(maze_file, num_mazes)
    print("Maze loading finished.")
    print()

    os.makedirs(png_dir, exist_ok=True)

    png_count = min(num_png, num_mazes)

    for i in range(png_count):
        save_path = os.path.join(png_dir, f"maze_{i:05d}.png")

        save_maze_png(
            maze=mazes[i],
            save_path=save_path,
            title=f"Maze {i:05d}",
            show_index=show_index,
        )

        print(f"Saved: {save_path}")

    print()
    print("Maze PNG export finished.")


def main():
    export_maze_png(
        maze_file="data/mazes_10000_seed0.bin",
        num_mazes=10000,
        num_png=20,
        png_dir="data/maze_png",
        show_index=False,
    )


if __name__ == "__main__":
    main()