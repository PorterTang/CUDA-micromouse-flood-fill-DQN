import numpy as np

from simulator.maze import (
    Maze,
    SIZE_MAZE,
    EAST,
    NORTH,
    SOUTH,
    WEST,
    DELTA_DIRECTION,
    grid_to_idx,
    idx_to_grid,
    in_bounds,
    turn_left,
    turn_right,
    turn_back,
)

from simulator.flood_fill import flood_fill, default_goal_indices, INF


class MicromouseEnv:
    """
    Micromouse environment

    Action:
        0: move forward
        1: turn left
        2: turn right
        3: turn back

    Observation dimension:
        12
    """

    def __init__(self, true_maze, max_steps=512):
        self.true_maze = true_maze
        self.size = true_maze.size
        self.num_cells = true_maze.num_cells
        self.max_steps = max_steps

        self.action_dim = 4
        self.obs_dim = 12

        self.goal_indices = default_goal_indices(self.size)
        self.goal_set = set(self.goal_indices)

        self.reset()

    def current_location(self):
        """
        Converting current grid loaction into idx
        """
        return grid_to_idx(self.row, self.col, self.size)

    def reset(self):
        """
        Reseting environment
        """
        self.row = 0
        self.col = 0
        self.direction = EAST

        self.steps = 0
        self.done = False

        # The maze that agent has explored 
        self.known_maze = Maze(self.size)

        # visited / discovered
        self.visited_count = np.zeros(self.num_cells, dtype=np.int32)
        self.discovered = np.zeros(self.num_cells, dtype=np.uint8)

        self.visited_count[self.current_location()] += 1

        self._sense_current_cell()

        return self._get_obs()

    def _sense_current_cell(self):
        """
        Sensing the wall info of current cell

        """
        idx_current = self.current_location()

        for direction in [NORTH, EAST, SOUTH, WEST]:
            if self.true_maze.has_wall(idx_current, direction):
                self.known_maze.add_wall(idx_current, direction)

        self.discovered[idx_current] = 1

    def _relative_dir(self, rel: str):
        if rel == "front":
            return self.direction
        if rel == "left":
            return turn_left(self.direction)
        if rel == "right":
            return turn_right(self.direction)
        if rel == "back":
            return turn_back(self.direction)

        raise ValueError("Unknown relative direction")

    def _wall_relative(self, rel: str) -> int:
        """
        Returning whether there is a wall in front/left/right
        """
        direction = self._relative_dir(rel)
        return int(self.true_maze.has_wall(self.current_location(), direction))

    def _neighbor_idx_by_direction(self, direction: int):
        """
        Getting neighbor's idx, based on current location and direction
        """
        direction_row, direction_col = DELTA_DIRECTION[direction]
        row_next, col_next = self.row + direction_row, self.col + direction_col

        if not in_bounds(row_next, col_next, self.size):
            return None

        return grid_to_idx(row_next, col_next, self.size)

    def _safe_distance_value(self, distance_record, direction: int) -> float:
        """
        Getting neighbor's Flood Fill distance，and normalizing to 0~1。
        """
        idx_current = self.current_location()

        if self.known_maze.has_wall(idx_current, direction):
            return 1.0

        idx_next = self._neighbor_idx_by_direction(direction)

        if idx_next is None:
            return 1.0

        distance = distance_record[idx_next]

        if distance >= INF:
            return 1.0

        return min(float(distance) / self.num_cells, 1.0)

    def _unknown_neighbor_count(self):
        """
        Counting unvisited neighbor
        """
        count = 0
        idx_current = self.current_location()

        for direction in [NORTH, EAST, SOUTH, WEST]:
            if self.true_maze.has_wall(idx_current, direction):
                continue

            idx_next = self._neighbor_idx_by_direction(direction)

            if idx_next is not None and self.discovered[idx_next] == 0:
                count += 1

        return count

    def _get_obs(self):
        """
        Building observation。

        """
        idx_current = self.current_location()

        distance = flood_fill(self.known_maze, self.goal_indices)

        distance_current = distance[idx_current]

        if distance_current >= INF:
            distance_current_norm = 1.0
        else:
            distance_current_norm = min(float(distance_current) / self.num_cells, 1.0)

        direction_front = self._relative_dir("front")
        direction_left = self._relative_dir("left")
        direction_right = self._relative_dir("right")

        obs = np.array(
            [
                self.row / (self.size - 1),
                self.col / (self.size - 1),
                self.direction / 3.0,

                self._wall_relative("front"),
                self._wall_relative("left"),
                self._wall_relative("right"),

                distance_current_norm,
                self._safe_distance_value(distance, direction_front),
                self._safe_distance_value(distance, direction_left),
                self._safe_distance_value(distance, direction_right),

                min(self.visited_count[idx_current] / 10.0, 1.0),
                self._unknown_neighbor_count() / 4.0,
            ],
            dtype=np.float32,
        )

        return obs

    def step(self, action: int):
        """
        Take an action。

        Reward:
            Each step: -1
            Hitting wall: -20
            New cell: +2
            flood distance decrease: +1
            flood distance increase: -1
            Revisiting: -0.5 * 次數
            Reach end point: +100
            Overtime: -50
        """
        if self.done:
            return self._get_obs(), 0.0, True, {}

        idx_current = self.current_location()

        distance_old_record = flood_fill(self.known_maze, self.goal_indices)
        distance_old = distance_old_record[idx_current]

        reward = -0.5
        collision = False

        if action == 1:
            self.direction = turn_left(self.direction)

        elif action == 2:
            self.direction = turn_right(self.direction)

        elif action == 3:
            self.direction = turn_back(self.direction)

        elif action == 0:
            if self.true_maze.has_wall(idx_current, self.direction):
                reward -= 30.0
                collision = True
            else:
                direction_row, direction_col = DELTA_DIRECTION[self.direction]
                row_next, col_next = self.row + direction_row, self.col + direction_col

                if not in_bounds(row_next, col_next, self.size):
                    reward -= 30.0
                    collision = True
                else:
                    self.row = row_next
                    self.col = col_next
        else:
            raise ValueError("Invalid action")

        self.steps += 1

        self._sense_current_cell()

        idx_new = self.current_location()

        if self.visited_count[idx_new] == 0:
            reward += 3.0

        self.visited_count[idx_new] += 1

        reward -= min(2.0, 0.2 * max(self.visited_count[idx_new] - 1, 0))

        distance_record_new = flood_fill(self.known_maze, self.goal_indices)
        distance_new = distance_record_new[idx_new]

        if distance_old < INF and distance_new < INF:
            if distance_new < distance_old:
                reward += 1.0
            elif distance_new > distance_old:
                reward -= 1.0

        success = idx_new in self.goal_set

        if success:
            reward += 200.0
            self.done = True

        if self.steps >= self.max_steps:
            reward -= 50.0
            self.done = True

        info = {
            "idx": idx_new,
            "row": self.row,
            "col": self.col,
            "direction": self.direction,
            "collision": collision,
            "steps": self.steps,
            "success": success,
        }

        return self._get_obs(), float(reward), self.done, info