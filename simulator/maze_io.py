"""
maze_io.py

Shared maze dataset I/O.

This module is used to make CPU, GPU, and RL use the same maze set.

Data format:
    Binary file stores uint8 wall values.

Shape:
    [num_mazes, 256]

Each maze:
    walls[256]

Each cell:
    4-bit wall encoding
"""

import os
import numpy as np

from simulator.maze import Maze, SIZE_MAZE, NUM_CELLS


def maze_to_array(maze: Maze) -> np.ndarray:
    """
    Convert Maze object to 1D uint8 array.

    Returns:
        walls array with shape [256]
    """
    return maze.flatten_walls().astype(np.uint8)


def array_to_maze(walls: np.ndarray, size: int = SIZE_MAZE) -> Maze:
    """
    Convert 1D wall array to Maze object.

    Args:
        walls:
            1D numpy array, shape = [256]
        size:
            maze size, default = 16

    Returns:
        Maze object
    """
    walls = np.asarray(walls, dtype=np.uint8)

    expected_cells = size * size

    if walls.shape[0] != expected_cells:
        raise ValueError(
            f"Invalid maze wall size: {walls.shape[0]}, "
            f"expected {expected_cells}"
        )

    maze = Maze(size)

    # Cover the internal walls of Maze.
    # Note: No need to regenerate the maze.
    maze.walls = walls.copy()

    return maze


def save_maze_batch(mazes, output_path: str):
    """
    Save a list of Maze objects to binary file.

    Args:
        mazes:
            list of Maze objects
        output_path:
            binary output path
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    num_mazes = len(mazes)

    walls_batch = np.zeros((num_mazes, NUM_CELLS), dtype=np.uint8)

    for idx, maze in enumerate(mazes):
        walls_batch[idx, :] = maze_to_array(maze)

    walls_batch.tofile(output_path)

    return walls_batch


def load_maze_batch(input_path: str, num_mazes: int, size: int = SIZE_MAZE):
    """
    Load maze batch from binary file.

    Args:
        input_path:
            binary file path
        num_mazes:
            number of mazes to load
        size:
            maze size

    Returns:
        walls_batch:
            numpy array, shape = [num_mazes, 256], dtype = uint8
    """
    num_cells = size * size
    expected_count = num_mazes * num_cells

    raw = np.fromfile(input_path, dtype=np.uint8)

    if raw.size != expected_count:
        raise ValueError(
            f"File size mismatch: {input_path}\n"
            f"Expected uint8 count: {expected_count}\n"
            f"Actual uint8 count: {raw.size}\n"
            f"Expected bytes: {expected_count}\n"
            f"Actual bytes: {raw.size}"
        )

    walls_batch = raw.reshape(num_mazes, num_cells)

    return walls_batch


def load_maze_objects(input_path: str, num_mazes: int, size: int = SIZE_MAZE):
    """
    Load maze batch and convert each maze to Maze object.

    This is mainly used by CPU benchmark and RL environment.
    """
    walls_batch = load_maze_batch(input_path, num_mazes, size)

    mazes = []

    for i in range(num_mazes):
        maze = array_to_maze(walls_batch[i], size)
        mazes.append(maze)

    return mazes


def check_maze_file(input_path: str, num_mazes: int, size: int = SIZE_MAZE):
    """
    Check whether the maze binary file exists and has correct size.
    """
    num_cells = size * size
    expected_bytes = num_mazes * num_cells

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Maze file not found: {input_path}")

    actual_bytes = os.path.getsize(input_path)

    if actual_bytes != expected_bytes:
        raise ValueError(
            f"Maze file size mismatch.\n"
            f"Path: {input_path}\n"
            f"Expected bytes: {expected_bytes}\n"
            f"Actual bytes: {actual_bytes}"
        )

    print("Maze file check passed.")
    print(f"Path: {input_path}")
    print(f"Number of mazes: {num_mazes}")
    print(f"Bytes: {actual_bytes}")