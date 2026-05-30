"""
1-dim DFS maze generator
Using recursive backtracking to generate perfect maze
"""

import random

from simulator.maze import (
    Maze,
    SIZE_MAZE,
    NORTH,
    EAST,
    SOUTH,
    WEST,
    DELTA_DIRECTION,
    in_bounds,
)


def generate_dfs_maze(size: int = SIZE_MAZE, seed=None) -> Maze:
    """
    Generating DFS maze。

    Process：
    1. Building cell that has wall on each side
    2. Init DFS from starting point
    3. Randomly selecting unvisited neighbor
    4. Removing wall between each cell
    """
    def grid_to_idx(row, col, size):
        return row * size + col

    def dfs(row: int, col: int):
        visited[grid_to_idx(row, col, size)] = True

        directions = [NORTH, EAST, SOUTH, WEST]
        random.shuffle(directions)

        for direction in directions:
            direction_row, direction_col = DELTA_DIRECTION[direction]
            row_next, col_next = row + direction_row, col + direction_col

            if not in_bounds(row_next, col_next, size):
                continue

            idx_next = grid_to_idx(row_next, col_next, size)

            if visited[idx_next]:
                continue
            
            idx_current = grid_to_idx(row, col, size)
            maze.remove_wall(idx_current, direction)
            dfs(row_next, col_next)

    if seed is not None:
        random.seed(seed)

    maze = Maze(size)

    # 先把每一格四面牆都補滿
    for row in range(size):
        for col in range(size):
            idx = grid_to_idx(row, col, size)
            for direction in [NORTH, EAST, SOUTH, WEST]:
                maze.add_wall(idx, direction)

    visited = [False for _ in range(size * size)]

    

    dfs(0, 0)

    return maze


def generate_batch_mazes(num_mazes: int, size: int = SIZE_MAZE, seed: int = 0):
    """
    產生多張 maze。
    """
    mazes = []

    for i in range(num_mazes):
        mazes.append(generate_dfs_maze(size=size, seed=seed + i))

    return mazes