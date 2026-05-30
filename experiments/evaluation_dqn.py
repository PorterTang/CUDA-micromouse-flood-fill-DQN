"""
evaluation_dqn.py

Evaluate trained DQN on shared maze dataset.

Run:

    python -m experiments.evaluation_dqn
"""

import os
import time
import numpy as np

from simulator.maze_io import load_maze_objects, check_maze_file
from simulator.micromouse_env import MicromouseEnv
from simulator.visualization import draw_maze, save_path_gif

from rl.dqn_agent import DQNAgent


def evaluate(num_eval_mazes=10000,
    max_steps=512,
    maze_file="data/mazes_10000_seed0.bin",
    model_path="results/dqn_micromouse_shared_mazes.pth",
    ):
    
    os.makedirs("results", exist_ok=True)

    print("DQN Evaluation with Shared Maze Dataset")
    print("---------------------------------------")
    print(f"Maze file: {maze_file}")
    print(f"Model path: {model_path}")
    print(f"Number of evaluation mazes: {num_eval_mazes}")

    check_maze_file(maze_file, num_eval_mazes)

    print()
    print("Loading shared mazes...")
    mazes = load_maze_objects(maze_file, num_eval_mazes)
    print("Maze loading finished.")
    
    env = MicromouseEnv(mazes[0], max_steps=max_steps)

    agent = DQNAgent(env.obs_dim, env.action_dim)
    agent.load(model_path)

    success_count = 0
    steps_list = []
    reward_list = []

    timer_start = time.perf_counter()
    
    for episode in range(num_eval_mazes):
        maze = mazes[episode]
        env = MicromouseEnv(maze, max_steps=max_steps)

        state = env.reset()
        path_indices = [env.current_location()]

        total_reward = 0.0
        info = {
            "success": False,
            "steps": 0,
        }

        for _ in range(max_steps):
            action = agent.select_action(state, epsilon=0.0)

            next_state, reward, done, info = env.step(action)

            state = next_state
            total_reward += reward
            path_indices.append(env.current_location())

            if done:
                break

        success = info["success"]

        if success:
            success_count += 1

        steps_list.append(info["steps"])
        reward_list.append(total_reward)

        print(
            f"Episode {episode + 1}: "
            f"Success={success}, "
            f"Steps={info['steps']}, "
            f"Reward={total_reward:.2f}"
        )

        if episode % 1000 == 0:
            draw_maze(
                maze,
                path_indices=path_indices,
                title="DQN Micromouse Evaluation Path - 1D",
                save_path=f"results/dqn_eval_path_1d_episode_{episode + 1}.png",
                show = False,
            )
            
            save_path_gif(
                maze=maze,
                path_indices=path_indices,
                gif_path=f"results/dqn_eval_path_shared_maze0_episode_{episode + 1}.gif",
                title="DQN Evaluation Path - Shared Maze[0]",
                step_duration=0.25,
                final_hold_frames=8,
            )

    timer_end = time.perf_counter()
    eval_time = timer_end - timer_start
    
    success_rate = success_count / num_eval_mazes * 100.0
    avg_steps = float(np.mean(steps_list))
    avg_reward = float(np.mean(reward_list))

    print()
    print("DQN Evaluation Result")
    print("---------------------")
    print(f"Evaluation time: {eval_time:.6f} sec")
    print(f"Average per maze: {eval_time / num_eval_mazes * 1000:.6f} ms")
    print(f"Success Rate: {success_rate:.2f}%")
    print(f"Average Steps: {avg_steps:.2f}")
    print(f"Average Reward: {avg_reward:.2f}")

    np.save("results/dqn_eval_steps_shared.npy", np.array(steps_list))
    np.save("results/dqn_eval_rewards_shared.npy", np.array(reward_list))

    summary_path = "results/dqn_eval_summary_shared.txt"

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("DQN Evaluation Result\n")
        f.write("---------------------\n")
        f.write(f"maze_file={maze_file}\n")
        f.write(f"model_path={model_path}\n")
        f.write(f"num_eval_mazes={num_eval_mazes}\n")
        f.write(f"evaluation_time_sec={eval_time:.6f}\n")
        f.write(f"avg_time_per_maze_ms={eval_time / num_eval_mazes * 1000:.6f}\n")
        f.write(f"success_rate={success_rate:.2f}\n")
        f.write(f"average_steps={avg_steps:.2f}\n")
        f.write(f"average_reward={avg_reward:.2f}\n")

    print(f"Summary saved to: {summary_path}")
    return 


def main():
    evaluate(num_eval_mazes=10000,
        max_steps=512,
        maze_file="data/mazes_10000_seed0.bin",
        model_path="results/dqn_micromouse_shared_mazes.pth",
        )
    return

if __name__ == "__main__":
    main()