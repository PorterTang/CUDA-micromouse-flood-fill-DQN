# DQN Training and Evaluation

## Overview

This project uses a DQN agent to learn Micromouse maze exploration.

The environment is implemented in:

```text
simulator/micromouse_env.py
```

The DQN agent is implemented in:

```text
rl/dqn_agent.py
```

Training script:

```text
rl/training_dqn.py
```

Evaluation script:

```text
experiments/evaluation_dqn.py
```

## Environment

The Micromouse environment uses:

```text
Action dimension = 4
Observation dimension = 12
```

Actions:

| Action |	Meaning |
|------|------|
| 0 |	Move forward |
| 1 |	Turn left |
| 2 |	Turn right |
| 3 |	Turn back |

## Observation

The observation includes:

```text
row position
col position
current direction
front wall
left wall
right wall
current flood-fill distance
neighbor flood-fill distance
visited count
unknown neighbor count
```

## Reward Design

The reward encourages:

```text
reaching the goal
moving toward lower flood-fill distance
visiting new cells
avoiding repeated visits
avoiding wall collisions
finishing before timeout
```

Typical reward components:

```text
step penalty
collision penalty
new cell reward
distance improvement reward
goal reward
timeout penalty
```

## Training

Run:

```bash
python -m rl.training_dqn
```

Outputs:

```text
results/dqn_micromouse_shared_mazes.pth
results/dqn_rewards_shared.npy
results/dqn_losses_shared.npy
results/dqn_success_shared.npy
results/dqn_steps_shared.npy
results/dqn_reward_curve_shared.png
```

## Evaluation

Run:

```bash
python -m experiments.evaluation_dqn
```

Outputs:

```text
results/dqn_eval_summary_shared.txt
results/dqn_eval_steps_shared.npy
results/dqn_eval_rewards_shared.npy
results/dqn_eval_path_shared_maze0.png
results/dqn_eval_path_shared_maze0.gif
```

## Evaluation Metrics

The evaluation summary includes:

```text
success_rate
average_steps
average_reward
evaluation_time_sec
avg_time_per_maze_ms
```

## DQN vs Flood Fill

Flood Fill computes a shortest path using full maze information.

DQN learns a policy through interaction with the environment.

Difference:

```text
Flood Fill path:
    Reconstructed from distance map.

DQN path:
    Actual agent trajectory.
```

## Notes

Training directly on 10,000 random mazes can be difficult.

Recommended debug strategy:

```text
1. Train on fixed maze first.
2. Verify the agent can overfit one maze.
3. Train on a small maze subset.
4. Expand to full shared dataset.
```
