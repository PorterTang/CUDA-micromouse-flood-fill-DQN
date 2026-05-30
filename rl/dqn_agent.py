"""
dqn_agent.py

DQN Agent。

"""

import random
import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    """
    MLP Q-network。

    Input:
        obs_dim

    Output:
        action_dim  Q-value number
    """

    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.ReLU(),

            nn.Linear(128, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        device: str = None,
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"[DQNAgent] Using device: {self.device}")

        self.q_net = QNetwork(obs_dim, action_dim).to(self.device)
        self.target_net = QNetwork(obs_dim, action_dim).to(self.device)

        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def select_action(self, state, epsilon: float):
        """
        epsilon-greedy action selection。
        """
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)

        state_tensor = torch.tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():
            q_values = self.q_net(state_tensor)

        return int(torch.argmax(q_values, dim=1).item())

    def update(self, replay_buffer, batch_size: int):
        """
        DQN update。

        target = reward + gamma * max Q_target(next_state)
        """
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_values = self.q_net(states).gather(1, actions)

        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(dim=1, keepdim=True)[0]
            targets = rewards + self.gamma * next_q_values * (1.0 - dones)

        loss = self.loss_fn(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)

        self.optimizer.step()

        return float(loss.item())

    def update_target(self):
        """
        Update target network。
        """
        self.target_net.load_state_dict(self.q_net.state_dict())

    def save(self, path: str):
        torch.save(self.q_net.state_dict(), path)

    def load(self, path: str):
        self.q_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.q_net.state_dict())