"""
1-dim array CPU Flood Fill
"""

from collections import deque
import numpy as np
from simulator.maze import (
    SIZE_MAZE,
    NUM_CELLS,
    NORTH,
    EAST,
    SOUTH,
    WEST,
    DELTA_DIRECTION,
    grid_to_idx,
    idx_to_grid,
    in_bounds,
)

INF = 10**9

def default_goal_indices(size: int = SIZE_MAZE):
    """
    Returning center cells' idx。

    For 16x16 maze：
        (7,7), (7,8), (8,7), (8,8)
    """
    mid1 = size // 2 - 1
    mid2 = size // 2

    return [
        grid_to_idx(mid1, mid1, size),
        grid_to_idx(mid1, mid2, size),
        grid_to_idx(mid2, mid1, size),
        grid_to_idx(mid2, mid2, size),
    ]


def default_goals(size: int = SIZE_MAZE):
    """
    回傳中心四格的 row, col。
    """
    mid1 = size // 2 - 1
    mid2 = size // 2

    return [
        (mid1, mid1),
        (mid1, mid2),
        (mid2, mid1),
        (mid2, mid2),
    ]


def flood_fill(maze, goal_indices=None):
    """
    1-dim array Flood Fill

    Args:
        maze: Maze object
        goal_indices: goal idx list

    Returns:
        distance: shape = [num_cells] 
    """
    size = maze.size
    num_cells = size * size

    if goal_indices is None:
        goal_indices = default_goal_indices(size)

    distance = np.full(num_cells, INF, dtype=np.int32)
    queue = deque()

    for idx_goal in goal_indices:
        distance[idx_goal] = 0
        queue.append(idx_goal)

    while queue:
        idx_current = queue.popleft()
        row, col = idx_to_grid(idx_current, size)

        for direction in [NORTH, EAST, SOUTH, WEST]:
            if maze.has_wall(idx_current, direction):
                continue

            direction_row, direction_col = DELTA_DIRECTION[direction]
            row_next, col_next = row + direction_row, col + direction_col

            if not in_bounds(row_next, col_next, size):
                continue

            idx_next = grid_to_idx(row_next, col_next, size)

            if distance[idx_next] > distance[idx_current] + 1:
                distance[idx_next] = distance[idx_current] + 1
                queue.append(idx_next)

    return distance


def reconstruct_path(maze, start_idx=0, goal_indices=None):
    """
    Rebuilding path based on distance map 

    Args:
        maze: Maze object
        start_idx: preset 0 -> (0,0)
        goal_indices: goal idx list

    Returns:
        path_indices: [idx0, idx1, ...]
        distance: distance array
    """
    if goal_indices is None:
        goal_indices = default_goal_indices(maze.size)

    goal_set = set(goal_indices)

    distance = flood_fill(maze, goal_indices)

    path = [start_idx]
    idx_current = start_idx

    max_steps = maze.num_cells * 4

    for _ in range(max_steps):
        if idx_current in goal_set:
            break

        idx_bext = None
        distance_bext = INF

        for idx_next, _ in maze.get_neighbors_idx(idx_current):
            if distance[idx_next] < distance_bext:
                distance_bext = distance[idx_next]
                idx_bext = idx_next

        if idx_bext is None:
            break

        idx_current = idx_bext
        path.append(idx_current)

    return path, distance


def path_indices_to_grid(path_indices, size: int = SIZE_MAZE):
    """
    Converting 1-dim array(idx) into 2-dim array(grid)
    Making plotting maze easier
    """
    return [idx_to_grid(idx, size) for idx in path_indices]


def print_distance_map(distance_record, size: int = SIZE_MAZE):
    """
    將一維 distance 印成 16x16 形式。
    """
    for row in range(size):
        row_str = []

        for col in range(size):
            idx = grid_to_idx(row, col, size)
            distance = distance_record[idx]

            if distance >= INF:
                row_str.append("##")
            else:
                row_str.append(f"{distance:02d}")

        print(" ".join(row_str))
        return