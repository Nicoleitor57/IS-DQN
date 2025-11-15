"""
IS_tigers_4.py

Double DQN + n-step returns + slower epsilon decay variant.

- Double DQN: select next action with local q_net and evaluate with target_q_net.
- n-step returns: accumulate n-step returns before storing to replay buffer.
- Slower epsilon decay: eps_start=0.8, eps_decay=0.9995 by default.

The script accepts command-line args for quick smoke runs (--episodes 1).
"""

import argparse
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import trange

from Entornos.TwoTigersEnv import TwoTigersEnv
from IS.DQN import QNet, ReplayBuffer


def check_gpu():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class DoubleDQNAgent:
    def __init__(self, state_space: int, action_space: int, device='cpu', lr=1e-3, gamma=0.99, batch_size=64, buffer_size=100_000):
        self.device = device
        self.gamma = gamma
        self.batch_size = batch_size
        self.action_space = action_space

        self.q_net = QNet(state_space, action_space).to(device)
        self.target_q_net = QNet(state_space, action_space).to(device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(obs_dim=state_space, size=buffer_size, batch_size=batch_size)

    def update(self):
        """Perform Double DQN update from replay buffer."""
        if len(self.replay_buffer) < self.batch_size:
            return None

        samples = self.replay_buffer.sample()
        states = torch.FloatTensor(samples['obs']).to(self.device)
        actions = torch.LongTensor(samples['acts'].reshape(-1,1)).to(self.device)
        rewards = torch.FloatTensor(samples['rews'].reshape(-1,1)).to(self.device)
        next_states = torch.FloatTensor(samples['next_obs']).to(self.device)
        dones = torch.FloatTensor(samples['done'].reshape(-1,1)).to(self.device)

        # Double DQN target calculation
        # next_actions according to current network
        with torch.no_grad():
            next_actions = self.q_net(next_states).argmax(1).unsqueeze(1)
            q_target_next = self.target_q_net(next_states).gather(1, next_actions)
            targets = rewards + (self.gamma * q_target_next * (1 - dones))

        q_out = self.q_net(states)
        q_a = q_out.gather(1, actions)
        loss = F.smooth_l1_loss(q_a, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def soft_update(self, tau=1e-2):
        for target_param, local_param in zip(self.target_q_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0 - tau)*target_param.data)


def main(episodes=200, max_steps=1000, n_step=3, lr=1e-3):
    device = check_gpu()
    env = TwoTigersEnv(max_episode_steps=max_steps)
    action_space = env.action_space_()
    state_space = env.state_space_()

    # Agent with Double DQN
    agent = DoubleDQNAgent(state_space, action_space, device=device, lr=lr, gamma=0.99, batch_size=64)

    # n-step buffer per episode
    n = n_step

    # Epsilon schedule (slower decay)
    eps_start = 0.8
    eps_end = 0.01
    eps_decay = 0.9995

    for episode in trange(episodes):
        epsilon = max(eps_end, eps_start * (eps_decay ** episode))

        obs, info = env.reset()
        p_t = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)

        total_reward = 0
        steps = 0
        losses = []

        # deque to store transitions for n-step: each item (state, action, reward, next_state, done)
        n_deque = deque()

        while True:
            # choose action from current belief state
            state_tensor = torch.FloatTensor(p_t).to(device)
            action_int = agent.q_net.sample_action(state_tensor, epsilon=epsilon)

            next_obs, reward, done, truncated, info = env.step(action_int)
            episode_done = done or truncated

            # update belief state (simple oracle as before)
            act1 = action_int // 3
            act2 = action_int % 3
            a_tuple = ("AL", "AR", "AE")[act1], ("AL", "AR", "AE")[act2]
            obs_map = {0: None, 1: "OL", 2: "OR"}
            o_tuple = (obs_map[next_obs[0]], obs_map[next_obs[1]])

            p1 = p_t[0:2]
            p2 = p_t[2:4]
            if a_tuple[0] == "AE":
                if o_tuple[0] == "OL":
                    numL = 0.85*p1[0]
                    numR = 0.15*p1[1]
                else:
                    numL = 0.15*p1[0]
                    numR = 0.85*p1[1]
                tot = numL + numR
                p1_next = np.array([numL/tot, numR/tot])
            else:
                p1_next = np.array([0.5, 0.5])

            if a_tuple[1] == "AE":
                if o_tuple[1] == "OL":
                    numL = 0.85*p2[0]
                    numR = 0.15*p2[1]
                else:
                    numL = 0.15*p2[0]
                    numR = 0.85*p2[1]
                tot = numL + numR
                p2_next = np.array([numL/tot, numR/tot])
            else:
                p2_next = np.array([0.5, 0.5])

            p_t_next = np.concatenate([p1_next, p2_next]).astype(np.float32)

            # push transition into n-step buffer
            n_deque.append((p_t.copy(), action_int, reward, p_t_next.copy(), episode_done))

            # when deque is large enough, form n-step transition for the oldest
            if len(n_deque) >= n:
                # compute n-step return for oldest
                R = 0.0
                for idx in range(n):
                    R += (agent.gamma ** idx) * n_deque[idx][2]
                state_0, action_0, _, _, _ = n_deque[0]
                next_state_n = n_deque[-1][3]
                done_n = n_deque[-1][4]
                # store aggregated transition
                agent.replay_buffer.put(state_0, action_0, R, next_state_n, done_n)
                n_deque.popleft()

            # if episode ended, flush remaining with truncated n smaller
            if episode_done:
                # flush
                while len(n_deque) > 0:
                    R = 0.0
                    for idx in range(len(n_deque)):
                        R += (agent.gamma ** idx) * n_deque[idx][2]
                    state_0, action_0, _, _, _ = n_deque[0]
                    next_state_n = n_deque[-1][3]
                    done_n = n_deque[-1][4]
                    agent.replay_buffer.put(state_0, action_0, R, next_state_n, done_n)
                    n_deque.popleft()

            # Training step
            if len(agent.replay_buffer) >= agent.batch_size:
                loss = agent.update()
                if loss is not None:
                    losses.append(loss)
                if steps % 5 == 0:
                    agent.soft_update(tau=1e-2)

            total_reward += reward
            steps += 1
            p_t = p_t_next

            if episode_done:
                break

        avg_loss = np.mean(losses) if len(losses) > 0 else 0.0
        print(f"Ep {episode+1} | Reward: {total_reward} | Steps: {steps} | AvgLoss: {avg_loss:.4f} | Eps: {epsilon:.4f}")

    print("Training finished")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=200)
    parser.add_argument('--max-steps', type=int, default=1000)
    parser.add_argument('--n-step', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()
    main(episodes=args.episodes, max_steps=args.max_steps, n_step=args.n_step, lr=args.lr)
