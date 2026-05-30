import os

from simulator.maze_generator import generate_dfs_maze
from simulator.flood_fill import reconstruct_path, print_distance_map
from simulator.visualization import draw_maze


def main():
    os.makedirs("results", exist_ok=True)

    maze = generate_dfs_maze(seed=42)

    path_indices, distance = reconstruct_path(maze, start_idx=0)

    print("Flood Fill Distance Map:")
    print_distance_map(distance, maze.size)

    print()
    print(f"Path length: {len(path_indices)}")
    print("Path indices:")
    print(path_indices)

    draw_maze(
        maze,
        path_indices=path_indices,
        distance=None,
        title="CPU Flood Fill Path - 1D Array",
        save_path="results/cpu_flood_fill_path_1d.png",
    )


if __name__ == "__main__":
    main()