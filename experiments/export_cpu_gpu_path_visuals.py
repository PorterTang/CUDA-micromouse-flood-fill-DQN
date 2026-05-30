"""
export_cpu_gpu_path_visuals.py

Use CPU/GPU Flood Fill distance binary outputs to reconstruct a path
from the start cell to the goal, then export PNG and GIF visualizations.

This script reads:

    data/mazes_10000_seed0.bin
    results/cpu_distances_10000.bin
    results/gpu_distances_10000.bin

and exports:

    results/cpu_flood_fill_path_shared_maze0.png
    results/cpu_flood_fill_path_shared_maze0.gif
    results/gpu_flood_fill_path_shared_maze0.png
    results/gpu_flood_fill_path_shared_maze0.gif

Optionally, it also exports distance-overlay PNG files.

Run:
    python -m experiments.export_cpu_gpu_path_visuals
"""

import os
import numpy as np

from simulator.maze import NORTH, EAST, SOUTH, WEST
from simulator.maze_io import load_maze_objects, check_maze_file
from simulator.visualization import draw_maze, save_path_gif, save_maze_png


INF_THRESHOLD = 10**8


def load_distance_batch(distance_file: str, num_mazes: int, size: int = 16):
    """
    Load distance binary file and reshape to [num_mazes, size*size].
    """
    if not os.path.exists(distance_file):
        raise FileNotFoundError(
            f"Distance file not found: {distance_file}"
        )

    expected_count = num_mazes * size * size
    expected_bytes = expected_count * np.dtype(np.int32).itemsize
    actual_bytes = os.path.getsize(distance_file)

    if actual_bytes != expected_bytes:
        raise ValueError(
            f"Distance file size mismatch.\n"
            f"Path: {distance_file}\n"
            f"Expected bytes: {expected_bytes}\n"
            f"Actual bytes: {actual_bytes}"
        )

    data = np.fromfile(distance_file, dtype=np.int32)

    if data.size != expected_count:
        raise ValueError(
            f"Distance file element count mismatch.\n"
            f"Path: {distance_file}\n"
            f"Expected count: {expected_count}\n"
            f"Actual count: {data.size}"
        )

    return data.reshape(num_mazes, size * size)


def get_goal_indices(size: int = 16):
    """
    Return the 4 center goal cell indices for Micromouse.
    """
    mid1 = size // 2 - 1
    mid2 = size // 2

    return {
        mid1 * size + mid1,
        mid1 * size + mid2,
        mid2 * size + mid1,
        mid2 * size + mid2,
    }


def get_neighbors(maze, idx: int):
    """
    Return all legal neighboring cells based on maze walls.

    Returns:
        list of (neighbor_idx, direction)
    """
    size = maze.size
    row = idx // size
    col = idx % size

    neighbors = []

    if row > 0 and not maze.has_wall(idx, NORTH):
        neighbors.append(((row - 1) * size + col, NORTH))

    if col < size - 1 and not maze.has_wall(idx, EAST):
        neighbors.append((row * size + (col + 1), EAST))

    if row < size - 1 and not maze.has_wall(idx, SOUTH):
        neighbors.append((((row + 1) * size + col), SOUTH))

    if col > 0 and not maze.has_wall(idx, WEST):
        neighbors.append((row * size + (col - 1), WEST))

    return neighbors


def reconstruct_path_from_distance(
    maze,
    distance_1d: np.ndarray,
    start_idx: int = 0,
):
    """
    Reconstruct a Flood Fill path from start to goal by greedily following
    a strictly smaller distance neighbor.

    Notes:
        - This is not an RL exploration trajectory.
        - This is a shortest-path style route implied by the distance map.

    Returns:
        path_indices: list of cell indices
    """
    size = maze.size
    goal_set = get_goal_indices(size)

    distance_1d = np.asarray(distance_1d, dtype=np.int32)

    if distance_1d.shape[0] != size * size:
        raise ValueError(
            f"Invalid distance shape: {distance_1d.shape}, "
            f"expected ({size * size},)"
        )

    if distance_1d[start_idx] >= INF_THRESHOLD:
        raise ValueError(
            f"Start cell distance is INF/unreachable: idx={start_idx}, "
            f"value={distance_1d[start_idx]}"
        )

    current = start_idx
    path_indices = [current]
    visited = {current}

    # Upper bound to avoid infinite loop
    max_path_len = size * size * 4

    for _ in range(max_path_len):
        if current in goal_set:
            return path_indices

        current_distance = int(distance_1d[current])
        neighbors = get_neighbors(maze, current)

        # Choose the legal neighbor with smallest distance,
        # preferably strictly smaller than current distance.
        best_next = None
        best_dist = current_distance

        for neighbor_idx, _ in neighbors:
            neighbor_dist = int(distance_1d[neighbor_idx])

            if neighbor_dist < best_dist:
                best_dist = neighbor_dist
                best_next = neighbor_idx

        if best_next is None:
            # No strictly smaller neighbor found.
            # Fallback: choose the smallest legal neighbor not yet visited.
            fallback_next = None
            fallback_dist = INF_THRESHOLD

            for neighbor_idx, _ in neighbors:
                neighbor_dist = int(distance_1d[neighbor_idx])

                if neighbor_idx not in visited and neighbor_dist < fallback_dist:
                    fallback_dist = neighbor_dist
                    fallback_next = neighbor_idx

            if fallback_next is None:
                raise RuntimeError(
                    f"Failed to reconstruct path.\n"
                    f"Current idx={current}, distance={current_distance}\n"
                    f"No smaller legal neighbor found."
                )

            best_next = fallback_next

        current = best_next
        path_indices.append(current)

        if current in visited and current not in goal_set:
            raise RuntimeError(
                f"Loop detected while reconstructing path.\n"
                f"Current idx={current}, distance={distance_1d[current]}"
            )

        visited.add(current)

    raise RuntimeError("Path reconstruction exceeded maximum allowed length.")


def export_one_visual_set(
    maze,
    distance_1d: np.ndarray,
    prefix: str,
    output_dir: str = "results",
    save_distance_overlay: bool = True,
):
    """
    Export one set of visuals:
        - path PNG
        - path GIF
        - optional distance-overlay PNG
    """
    os.makedirs(output_dir, exist_ok=True)

    path_indices = reconstruct_path_from_distance(
        maze=maze,
        distance_1d=distance_1d,
        start_idx=0,
    )

    path_png = os.path.join(output_dir, f"{prefix}.png")
    path_gif = os.path.join(output_dir, f"{prefix}.gif")

    draw_maze(
        maze=maze,
        path_indices=path_indices,
        title=prefix.replace("_", " "),
        save_path=path_png,
        show=False,
    )

    save_path_gif(
        maze=maze,
        path_indices=path_indices,
        gif_path=path_gif,
        title=prefix.replace("_", " "),
        step_duration=0.25,
        final_hold_frames=8,
    )

    print(f"Saved path PNG: {path_png}")
    print(f"Saved path GIF: {path_gif}")

    if save_distance_overlay:
        distance_png = os.path.join(output_dir, f"{prefix}_distance.png")

        save_maze_png(
            maze=maze,
            save_path=distance_png,
            distance=distance_1d,
            title=f"{prefix.replace('_', ' ')} distance map",
            show_index=False,
        )

        print(f"Saved distance PNG: {distance_png}")

    return path_indices


def export_cpu_gpu_visuals(
    maze_file="data/mazes_10000_seed0.bin",
    cpu_distance_file="results/cpu_distances_10000.bin",
    gpu_distance_file="results/gpu_distances_10000.bin",
    num_mazes=10000,
    maze_index=0,
    output_dir="results",
):
    """
    Export CPU/GPU Flood Fill path PNG/GIF visualizations using distance bins.
    """

    print("Export CPU/GPU Flood Fill Path Visuals")
    print("--------------------------------------")
    print(f"Maze file: {maze_file}")
    print(f"CPU distance file: {cpu_distance_file}")
    print(f"GPU distance file: {gpu_distance_file}")
    print(f"Number of mazes: {num_mazes}")
    print(f"Selected maze index: {maze_index}")
    print(f"Output directory: {output_dir}")
    print()

    if maze_index < 0 or maze_index >= num_mazes:
        raise ValueError(
            f"maze_index out of range: {maze_index}, "
            f"valid range = [0, {num_mazes - 1}]"
        )

    check_maze_file(maze_file, num_mazes)

    print("Loading maze objects...")
    mazes = load_maze_objects(maze_file, num_mazes)
    print("Maze loading finished.")

    print("Loading CPU distance batch...")
    cpu_distances = load_distance_batch(cpu_distance_file, num_mazes, size=16)

    print("Loading GPU distance batch...")
    gpu_distances = load_distance_batch(gpu_distance_file, num_mazes, size=16)

    maze = mazes[maze_index]
    cpu_distance_1d = cpu_distances[maze_index]
    gpu_distance_1d = gpu_distances[maze_index]

    print()
    print("Exporting CPU visuals...")
    cpu_path = export_one_visual_set(
        maze=maze,
        distance_1d=cpu_distance_1d,
        prefix=f"cpu_flood_fill_path_shared_maze{maze_index}",
        output_dir=output_dir,
        save_distance_overlay=True,
    )

    print()
    print("Exporting GPU visuals...")
    gpu_path = export_one_visual_set(
        maze=maze,
        distance_1d=gpu_distance_1d,
        prefix=f"gpu_flood_fill_path_shared_maze{maze_index}",
        output_dir=output_dir,
        save_distance_overlay=True,
    )

    same_path = cpu_path == gpu_path
    same_distance = np.array_equal(cpu_distance_1d, gpu_distance_1d)

    summary_path = os.path.join(
        output_dir,
        f"cpu_gpu_path_visual_summary_maze{maze_index}.txt",
    )

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("CPU/GPU Path Visual Export Summary\n")
        f.write("----------------------------------\n")
        f.write(f"maze_file={maze_file}\n")
        f.write(f"cpu_distance_file={cpu_distance_file}\n")
        f.write(f"gpu_distance_file={gpu_distance_file}\n")
        f.write(f"num_mazes={num_mazes}\n")
        f.write(f"maze_index={maze_index}\n")
        f.write(f"cpu_distance_equal_gpu_distance={same_distance}\n")
        f.write(f"cpu_path_equal_gpu_path={same_path}\n")
        f.write(f"cpu_path_length={len(cpu_path)}\n")
        f.write(f"gpu_path_length={len(gpu_path)}\n")
        f.write(
            f"cpu_png={os.path.join(output_dir, f'cpu_flood_fill_path_shared_maze{maze_index}.png')}\n"
        )
        f.write(
            f"cpu_gif={os.path.join(output_dir, f'cpu_flood_fill_path_shared_maze{maze_index}.gif')}\n"
        )
        f.write(
            f"gpu_png={os.path.join(output_dir, f'gpu_flood_fill_path_shared_maze{maze_index}.png')}\n"
        )
        f.write(
            f"gpu_gif={os.path.join(output_dir, f'gpu_flood_fill_path_shared_maze{maze_index}.gif')}\n"
        )

    print()
    print(f"Summary saved to: {summary_path}")
    print(f"CPU distance == GPU distance: {same_distance}")
    print(f"CPU path == GPU path: {same_path}")


def main():
    export_cpu_gpu_visuals(
        maze_file="data/mazes_10000_seed0.bin",
        cpu_distance_file="results/cpu_distances_10000.bin",
        gpu_distance_file="results/gpu_distances_10000.bin",
        num_mazes=10000,
        maze_index=0,
        output_dir="results",
    )


if __name__ == "__main__":
    main()