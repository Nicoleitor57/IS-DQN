import random
from typing import Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# -----------------------------
# Q-network
# -----------------------------
class QNet(nn.Module):
    def __init__(self, state_space: int, action_space: int):
        super(QNet, self).__init__()
        self.state_space = state_space
        self.action_space = action_space

        self.fc1 = nn.Linear(state_space, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_space)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def sample_action(self, obs, epsilon: float):
        """Epsilon-greedy action selection"""
        if random.random() < epsilon:
            return random.randint(0, self.action_space - 1)
        else:
            with torch.no_grad():
                return self.forward(obs).argmax().item()


# -----------------------------
# Replay Buffer
# -----------------------------
class ReplayBuffer:
    """A simple numpy-based replay buffer for DQN"""
    def __init__(self, obs_dim: int, size: int, batch_size: int = 32):
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size], dtype=np.float32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.max_size = size
        self.batch_size = batch_size
        self.ptr = 0
        self.size = 0

    def put(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self) -> Dict[str, np.ndarray]:
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
        return dict(
            obs=self.obs_buf[idxs],
            next_obs=self.next_obs_buf[idxs],
            acts=self.acts_buf[idxs],
            rews=self.rews_buf[idxs],
            done=self.done_buf[idxs]
        )

    def __len__(self):
        return self.size


# -----------------------------
# DQN Agent (Encapsulates training)
# -----------------------------
class DQNAgent:
    def __init__(self, state_space: int, action_space: int, device='cpu', lr=1e-3, gamma=0.99, batch_size=64):
        self.device = device
        self.gamma = gamma
        self.batch_size = batch_size

        # Q-networks
        self.q_net = QNet(state_space, action_space).to(device)
        self.target_q_net = QNet(state_space, action_space).to(device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(obs_dim=state_space, size=100_000, batch_size=batch_size)

    def update(self):
        """Train Q-network from replay buffer. Returns loss for logging."""
        if len(self.replay_buffer) < self.batch_size:
            return None  # Not enough samples yet

        samples = self.replay_buffer.sample()

        states = torch.FloatTensor(samples['obs']).to(self.device)
        actions = torch.LongTensor(samples['acts'].reshape(-1,1)).to(self.device)
        rewards = torch.FloatTensor(samples['rews'].reshape(-1,1)).to(self.device)
        next_states = torch.FloatTensor(samples['next_obs']).to(self.device)
        dones = torch.FloatTensor(samples['done'].reshape(-1,1)).to(self.device)

        # Compute target
        q_target_max = self.target_q_net(next_states).max(1)[0].unsqueeze(1).detach()
        targets = rewards + self.gamma * q_target_max * (1 - dones)

        q_out = self.q_net(states)
        q_a = q_out.gather(1, actions)

        loss = F.smooth_l1_loss(q_a, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()  # Return loss for monitoring

    def soft_update(self, tau=1e-2):
        """Soft update target network"""
        for target_param, local_param in zip(self.target_q_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0 - tau)*target_param.data)

    def save_model(self, path='dqn_model.pth'):
        torch.save(self.q_net.state_dict(), path)

    def load_model(self, path):
        self.q_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_q_net.load_state_dict(self.q_net.state_dict())
