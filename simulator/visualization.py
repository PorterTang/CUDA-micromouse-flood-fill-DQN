"""
visualization.py

Visualization utilities for Micromouse maze.

Features:
1. Save maze as PNG
2. Draw maze path
3. Draw Flood Fill distance map
4. Generate path GIF
5. Plot training curves

Goal area:
    The center four cells are highlighted with red background.
"""

import os
import tempfile

import imageio.v2 as imageio
import matplotlib.pyplot as plt

from simulator.maze import (
    NORTH,
    EAST,
    SOUTH,
    WEST,
    grid_to_idx,
    idx_to_grid,
)


def _draw_goal_background(ax, size):
    """
    Draw red background for Micromouse goal area.

    For 16x16 maze, goal cells are:
        (7,7), (7,8), (8,7), (8,8)
    """

    mid1 = size // 2 - 1
    mid2 = size // 2

    goal_cells = [
        (mid1, mid1),
        (mid1, mid2),
        (mid2, mid1),
        (mid2, mid2),
    ]

    for goal_row, goal_col in goal_cells:
        x_goal = goal_col
        y_goal = size - 1 - goal_row

        rect = plt.Rectangle(
            (x_goal, y_goal),
            1,
            1,
            facecolor="red",
            alpha=0.25,
            edgecolor=None,
        )

        ax.add_patch(rect)


def _draw_maze_walls_and_text(
    ax,
    maze,
    distance=None,
    show_index=False,
):
    """
    Draw maze walls and optional cell text.
    """

    size = maze.size

    for row in range(size):
        for col in range(size):
            idx = grid_to_idx(row, col, size)

            x = col
            y = size - 1 - row

            if maze.has_wall(idx, NORTH):
                ax.plot([x, x + 1], [y + 1, y + 1], linewidth=2)

            if maze.has_wall(idx, EAST):
                ax.plot([x + 1, x + 1], [y, y + 1], linewidth=2)

            if maze.has_wall(idx, SOUTH):
                ax.plot([x, x + 1], [y, y], linewidth=2)

            if maze.has_wall(idx, WEST):
                ax.plot([x, x], [y, y + 1], linewidth=2)

            if distance is not None:
                ax.text(
                    x + 0.5,
                    y + 0.5,
                    str(int(distance[idx])),
                    ha="center",
                    va="center",
                    fontsize=7,
                )

            elif show_index:
                ax.text(
                    x + 0.5,
                    y + 0.5,
                    str(idx),
                    ha="center",
                    va="center",
                    fontsize=6,
                )


def _draw_path(ax, path_indices, size):
    """
    Draw path on maze.
    """

    if path_indices is None or len(path_indices) <= 1:
        return

    xs = []
    ys = []

    for idx in path_indices:
        row, col = idx_to_grid(idx, size)

        xs.append(col + 0.5)
        ys.append(size - 1 - row + 0.5)

    ax.plot(xs, ys, marker="o", linewidth=2)

    last_idx = path_indices[-1]
    row, col = idx_to_grid(last_idx, size)

    ax.plot(
        col + 0.5,
        size - 1 - row + 0.5,
        marker="o",
        markersize=10,
    )


def _finalize_figure(ax, size, title):
    """
    Apply common figure settings.
    """

    ax.set_aspect("equal")
    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    ax.set_xticks([])
    ax.set_yticks([])

    if title is not None:
        ax.set_title(title)


def save_maze_png(
    maze,
    save_path,
    path_indices=None,
    distance=None,
    title=None,
    show_index=False,
    dpi=150,
):
    """
    Save maze as PNG.
    """

    output_dir = os.path.dirname(save_path)

    if output_dir != "":
        os.makedirs(output_dir, exist_ok=True)

    size = maze.size

    fig, ax = plt.subplots(figsize=(8, 8))

    _draw_goal_background(ax, size)
    _draw_maze_walls_and_text(
        ax=ax,
        maze=maze,
        distance=distance,
        show_index=show_index,
    )
    _draw_path(ax, path_indices, size)
    _finalize_figure(ax, size, title)

    plt.savefig(save_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)


def draw_maze(
    maze,
    path_indices=None,
    distance=None,
    title="Micromouse Maze",
    save_path=None,
    show=True,
    show_index=False,
):
    """
    Draw maze and optionally save it.
    """

    size = maze.size

    fig, ax = plt.subplots(figsize=(8, 8))

    _draw_goal_background(ax, size)
    _draw_maze_walls_and_text(
        ax=ax,
        maze=maze,
        distance=distance,
        show_index=show_index,
    )
    _draw_path(ax, path_indices, size)
    _finalize_figure(ax, size, title)

    if save_path is not None:
        output_dir = os.path.dirname(save_path)

        if output_dir != "":
            os.makedirs(output_dir, exist_ok=True)

        plt.savefig(save_path, bbox_inches="tight", dpi=150)

    if show:
        plt.show()
    else:
        plt.close(fig)


def draw_maze_frame(
    maze,
    current_path=None,
    distance=None,
    title="Micromouse Maze",
    save_path=None,
    show=False,
    show_index=False,
):
    """
    Draw one frame for GIF animation.
    """

    size = maze.size

    fig, ax = plt.subplots(figsize=(8, 8))

    _draw_goal_background(ax, size)
    _draw_maze_walls_and_text(
        ax=ax,
        maze=maze,
        distance=distance,
        show_index=show_index,
    )
    _draw_path(ax, current_path, size)
    _finalize_figure(ax, size, title)

    if save_path is not None:
        output_dir = os.path.dirname(save_path)

        if output_dir != "":
            os.makedirs(output_dir, exist_ok=True)

        plt.savefig(save_path, bbox_inches="tight", dpi=150)

    if show:
        plt.show()
    else:
        plt.close(fig)


def save_path_gif(
    maze,
    path_indices,
    gif_path,
    distance=None,
    title="Micromouse Path GIF",
    step_duration=0.25,
    final_hold_frames=8,
):
    """
    Save path animation as GIF.
    """

    output_dir = os.path.dirname(gif_path)

    if output_dir != "":
        os.makedirs(output_dir, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        frame_paths = []

        for step in range(1, len(path_indices) + 1):
            frame_path = os.path.join(tmpdir, f"frame_{step:04d}.png")

            draw_maze_frame(
                maze=maze,
                current_path=path_indices[:step],
                distance=distance,
                title=f"{title} - Step {step}/{len(path_indices)}",
                save_path=frame_path,
                show=False,
            )

            frame_paths.append(frame_path)

        images = [imageio.imread(frame_path) for frame_path in frame_paths]

        if len(images) > 0:
            for _ in range(final_hold_frames):
                images.append(images[-1])

        imageio.mimsave(gif_path, images, duration=step_duration)

    print(f"GIF saved to: {gif_path}")


def plot_training_curve(rewards, save_path=None, show=True):
    """
    Plot training reward curve.
    """

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(rewards)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Reward")
    ax.set_title("Training Reward Curve")
    ax.grid(True)

    if save_path is not None:
        output_dir = os.path.dirname(save_path)

        if output_dir != "":
            os.makedirs(output_dir, exist_ok=True)

        plt.savefig(save_path, bbox_inches="tight", dpi=150)

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_success_curve(success_history, window=50, save_path=None, show=True):
    """
    Plot moving average success rate.
    """

    if len(success_history) == 0:
        print("No success history to plot.")
        return

    moving_avg = []

    for i in range(len(success_history)):
        start = max(0, i - window + 1)
        avg = sum(success_history[start:i + 1]) / (i - start + 1)
        moving_avg.append(avg * 100.0)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(moving_avg)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Success Rate (%)")
    ax.set_title(f"Success Rate Moving Average (window={window})")
    ax.grid(True)

    if save_path is not None:
        output_dir = os.path.dirname(save_path)

        if output_dir != "":
            os.makedirs(output_dir, exist_ok=True)

        plt.savefig(save_path, bbox_inches="tight", dpi=150)

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_steps_curve(steps_history, window=50, save_path=None, show=True):
    """
    Plot moving average episode steps.
    """

    if len(steps_history) == 0:
        print("No steps history to plot.")
        return

    moving_avg = []

    for i in range(len(steps_history)):
        start = max(0, i - window + 1)
        avg = sum(steps_history[start:i + 1]) / (i - start + 1)
        moving_avg.append(avg)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(moving_avg)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Average Steps")
    ax.set_title(f"Steps Moving Average (window={window})")
    ax.grid(True)

    if save_path is not None:
        output_dir = os.path.dirname(save_path)

        if output_dir != "":
            os.makedirs(output_dir, exist_ok=True)

        plt.savefig(save_path, bbox_inches="tight", dpi=150)

    if show:
        plt.show()
    else:
        plt.close(fig)