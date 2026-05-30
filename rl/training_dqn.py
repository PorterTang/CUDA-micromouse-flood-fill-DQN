import os
import numpy as np
from tqdm import tqdm

from simulator.maze_io import load_maze_objects, check_maze_file
from simulator.micromouse_env import MicromouseEnv
from simulator.visualization import plot_training_curve

from rl.dqn_agent import DQNAgent
from rl.replay_buffer import ReplayBuffer

EPISODES = 5000
BATCH_SIZE = 128
BUFFER_CAPACITY = 100000
MIN_BUFFER_SIZE = 5000
TARGET_UPDATE_INTERVAL = 100
MAX_STEPS = 512
MAZE_FILE = "data/mazes_10000_seed0.bin"
NUM_MAZES = 10000
FIXED_MAZE = False

def train_dqn(
    episodes=5000,
    batch_size=128,
    buffer_capacity=100000,
    min_buffer_size=5000,
    target_update_interval=100,
    max_steps=512,
    maze_file="data/mazes_10000_seed0.bin",
    num_mazes=10000,
    fixed_maze=False,
):
    """
    Train DQN agent on shared maze dataset.

    Args:
        episodes:
            total training episodes
        batch_size:
            DQN mini-batch size
        buffer_capacity:
            replay buffer capacity
        min_buffer_size:
            start training only after replay buffer reaches this size
        target_update_interval:
            update target network every N episodes
        max_steps:
            max steps per episode
        maze_file:
            shared maze binary file
        num_mazes:
            number of mazes in shared dataset
        fixed_maze:
            if True, always use maze[0] for overfit debugging
    """
    
    os.makedirs("results", exist_ok=True)
    
    print("DQN Training with Shared Maze Dataset")
    print("-------------------------------------")
    print(f"Maze file: {maze_file}")
    print(f"Number of mazes: {num_mazes}")
    print(f"Episodes: {episodes}")
    print(f"Fixed maze mode: {fixed_maze}")

    check_maze_file(maze_file, num_mazes)

    print()
    print("Loading shared mazes...")
    mazes = load_maze_objects(maze_file, num_mazes)
    print("Maze loading finished.")

    env = MicromouseEnv(mazes[0], max_steps=max_steps)

    agent = DQNAgent(
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        lr=1e-4,
        gamma=0.99,
    )

    replay_buffer = ReplayBuffer(buffer_capacity)

    rewards_history = []
    loss_history = []
    success_history = []
    steps_history = []

    # Epsilon-greedy parameters
    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_decay = 1500

    for episode in tqdm(range(episodes), desc="Training DQN"):
        if fixed_maze:
            maze = mazes[0]
        else:
            maze = mazes[episode % num_mazes]
            
        env = MicromouseEnv(maze, max_steps=max_steps)

        state = env.reset()
        
        episode_reward = 0.0
        episode_losses = []

        epsilon =  epsilon_end + (epsilon_start - epsilon_end) * np.exp(-episode / epsilon_decay)

        info = {
            "success": False,
            "steps": 0,
        }
        
        for _ in range(max_steps):
            action = agent.select_action(state, epsilon)

            next_state, reward, done, info = env.step(action)

            replay_buffer.push(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward

            if len(replay_buffer) >= min_buffer_size:
                loss = agent.update(replay_buffer, batch_size)
                episode_losses.append(loss)

            if done:
                break

        rewards_history.append(episode_reward)
        success_history.append(1 if info.get("success", False) else 0)
        steps_history.append(info.get("steps", max_steps))
        
        if episode_losses:
            loss_history.append(float(np.mean(episode_losses)))
        else:
            loss_history.append(0.0)

        if episode % target_update_interval == 0:
            agent.update_target()

        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(rewards_history[-50:])
            avg_loss = np.mean(loss_history[-50:])

            print(
                f"\n",
                f"Episode {episode + 1}, "
                f"Avg Reward: {avg_reward:.2f}, "
                f"Avg Loss: {avg_loss:.4f}, "
                f"Epsilon: {epsilon:.3f}"
            )

    
    model_path = "results/dqn_micromouse_shared_mazes.pth"
    agent.save(model_path)

    np.save("results/dqn_rewards_shared.npy", np.array(rewards_history))
    np.save("results/dqn_losses_shared.npy", np.array(loss_history))
    np.save("results/dqn_success_shared.npy", np.array(success_history))
    np.save("results/dqn_steps_shared.npy", np.array(steps_history))

    plot_training_curve(
        rewards_history,
        save_path="results/dqn_reward_curve_shared.png",
    )

    print()
    print("Training finished.")
    print(f"Model saved to: {model_path}")
    return 


def main():
    
    train_dqn(
        episodes=EPISODES,
        batch_size=BATCH_SIZE,
        buffer_capacity=BUFFER_CAPACITY,
        min_buffer_size=MIN_BUFFER_SIZE,
        target_update_interval=TARGET_UPDATE_INTERVAL,
        max_steps=MAX_STEPS,
        maze_file=MAZE_FILE,
        num_mazes=NUM_MAZES,
        fixed_maze=FIXED_MAZE
    )
    
    return

if __name__ == "__main__":
    main()