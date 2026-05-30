"""
This program defines the structure of the maze.
In this project, we used 16 x 16 grid maze.
Each grid has 4 bits:
bit 0: North wall
bit 1: East wall
bit 2: South wall
bit 3: West wall 
"""
import numpy as np

# The size of maze
SIZE_MAZE = 16
NUM_CELLS = SIZE_MAZE * SIZE_MAZE

# Direction of movement
NORTH = 0
EAST = 1
SOUTH = 2
WEST = 3

# Corresponding wall on each direction
WALL_NORTH = 1 << 0
WALL_EAST = 1 << 1 
WALL_SOUTH = 1 << 2
WALL_WEST = 1 << 3

WALL_BITS = {
    NORTH: WALL_NORTH,
    EAST: WALL_EAST,
    SOUTH: WALL_SOUTH,
    WEST: WALL_WEST,
}

# Opposite direction
OPPOSITE_DIRECTION = {
    NORTH: SOUTH,
    EAST: WEST,
    SOUTH: NORTH,
    WEST: EAST,
}

# Moving size of each direction
DELTA_DIRECTION = {
    NORTH: (-1, 0),
    EAST: (0, 1),
    SOUTH: (1, 0),
    WEST: (0, -1),
}

def grid_to_idx(row: int, col: int, size: int = SIZE_MAZE):
    """
    Turning 2-dim array(grid) into 1-dim array(idx of cell)
    """
    return row * size + col

def idx_to_grid(idx: int, size:int = SIZE_MAZE):
    """
    Turning 1-dim array(idx of cell) into 2-dim array(grid)
    """
    return idx // size, idx % size

def in_bounds(row: int, col: int, size: int = SIZE_MAZE) -> bool:
    """
    Checking the coordinate is in the 16 x 16 maze or not
    """
    return 0 <= row < size and 0 <= col < size

def turn_left(direction: int) -> int:
    """
    Left turn 90 degree
    NORTH -> WEST
    WEST -> SOUTH
    SOUTH -> EAST
    EAST -> NORTH
    """
    return (direction - 1) % 4

def turn_right(direction: int) -> int:
    """
    Right turn 90 degree
    NORTH -> EAST
    WEST -> NORTH
    SOUTH -> WEST
    EAST -> SOUTH
    """
    return (direction + 1) % 4

def turn_back(direction: int) -> int:
    """
    Turn 180 degree
    NORTH -> SOUTH
    WEST -> EAST
    SOUTH -> NORTH
    EAST -> WEST
    """
    return (direction + 2) % 4

class Maze:
    """
    Class Maze is responsible for store the data structure of the maze
    self.walls[row, col] is a 4-bit int, represents the wall info on each direction
    """

    def __init__(self, size: int = SIZE_MAZE):
        self.size = size
        self.num_cells = size * size
        self.walls = np.zeros(self.num_cells, dtype=np.uint8)

        self._add_boundary_walls()
        return
    
    def idx(self, row: int, col: int):
        return grid_to_idx(row, col, self.size)
    
    def row_col(self, idx: int):
        return idx_to_grid(idx, self.size)
    
    def _add_boundary_walls(self):
        """
        Building outside walls
        """
        for col in range(self.size):
            idx_top = grid_to_idx(0, col, self.size)
            idx_bottom = grid_to_idx(self.size - 1, col, self.size)
            self.add_wall(idx_top, NORTH)
            self.add_wall(idx_bottom, SOUTH)

        for row in range(self.size):
            idx_left = grid_to_idx(row, 0, self.size)
            idx_right = grid_to_idx(row, self.size - 1, self.size)
            self.add_wall(idx_left, WEST)
            self.add_wall(idx_right, EAST)

        return
    
    def has_wall(self, idx: int, direction: int):
        return bool(self.walls[idx] & WALL_BITS[direction])
    
    def add_wall(self, idx: int, direction: int):
        row, col = idx_to_grid(idx, self.size)

        if not in_bounds(row, col, self.size):
            return
        
        self.walls[idx] |= WALL_BITS[direction]
        direction_row, direction_col = DELTA_DIRECTION[direction]
        row_next = row + direction_row
        col_next = col + direction_col

        if in_bounds(row_next, col_next, self.size):
            idx_next = grid_to_idx(row_next, col_next, self.size)
            opposite = OPPOSITE_DIRECTION[direction]
            self.walls[idx_next] |= WALL_BITS[opposite]

        return

    def remove_wall(self, idx: int, direction: int):
        row, col = idx_to_grid(idx, self.size)

        if not in_bounds(row, col, self.size):
            return

        mask = np.uint8(0xFF ^ WALL_BITS[direction])
        self.walls[idx] &= mask

        direction_row, direction_col = DELTA_DIRECTION[direction]
        row_next = row + direction_row
        col_next = col + direction_col

        if in_bounds(row_next, col_next, self.size):
            idx_next = grid_to_idx(row_next, col_next, self.size)
            opposite = OPPOSITE_DIRECTION[direction]
            opposite_mask = np.uint8(0xFF ^ WALL_BITS[opposite])
            self.walls[idx_next] &= opposite_mask

        return
    
    def get_neighbors_idx(self, idx: int):
        row, col = idx_to_grid(idx, self.size)

        neighbors = []

        for direction in [NORTH, EAST, SOUTH, WEST]:
            if self.has_wall(idx, direction):
                continue

            direction_row, direction_col = DELTA_DIRECTION[direction]
            row_next = row + direction_row
            col_next = col + direction_col

            if in_bounds(row_next, col_next, self.size):
                idx_next = grid_to_idx(row_next, col_next, self.size)
                neighbors.append((idx_next, direction))

        return neighbors
    
    def copy(self):
        new_maze = Maze(self.size)
        new_maze.walls = self.walls.copy()
        return new_maze

    def flatten_walls(self):
        return self.walls.copy()