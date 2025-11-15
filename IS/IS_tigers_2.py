"""
IS_tigers_2.py

Variant that implements a small learned PSR module and uses its predictions
as inductive bias/features for a DQN agent.

Assumptions (inferred):
- The paper's PSR describes predictive features for short tests; here we
  implement a lightweight learned predictor that maps current belief + last
  (action,obs) into predictions for a small set of tests.
- Tests used: probability that a "listen" (AE) on tiger 1/2 will produce "OL".
  This gives 2 predictive features (one per tiger). This is a lightweight
  approximation of the predictive-tests idea in PSR papers.

If you want the exact PSR parametrization from the PDF (matrix-based PSR,
learned via spectral methods), tell me and I can implement that next — it's
more involved but feasible.
"""

from Entornos.TwoTigersEnv import TwoTigersEnv
from .DQN import DQNAgent

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange
from torch.utils.tensorboard import SummaryWriter


class PSRPredictor(nn.Module):
    """Small network that predicts probabilities for a set of tests.

    Here we predict, for each tiger independently, the probability that a
    subsequent "listen" (AE) yields observation "OL" (i.e., sound left).
    This gives two outputs (one per tiger). The network is trained online
    from transitions where a listen was performed.
    """
    def __init__(self, input_dim, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
            nn.Sigmoid()  # probabilities in (0,1)
        )

    def forward(self, x):
        return self.net(x)


class PSRBuffer:
    """Small ring buffer to store recent (input, target) for PSR training.
    Targets are 0/1 for "OL" on tiger1 and tiger2 when a listen action occurred.
    """
    def __init__(self, input_dim, size=5000):
        self.inputs = np.zeros((size, input_dim), dtype=np.float32)
        self.targets = np.zeros((size, 2), dtype=np.float32)
        self.ptr = 0
        self.size = 0
        self.max_size = size

    def add(self, inp, target):
        self.inputs[self.ptr] = inp
        self.targets[self.ptr] = target
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch):
        if self.size < batch:
            idx = np.random.choice(self.size, size=self.size, replace=False)
        else:
            idx = np.random.choice(self.size, size=batch, replace=False)
        return self.inputs[idx], self.targets[idx]


def build_input_features(p_t, last_a, last_o):
    """Construct input vector for PSR predictor.

    - p_t: 4-dim belief vector ([pL1, pR1, pL2, pR2])
    - last_a: int action (0-8) or None
    - last_o: tuple observation (o1,o2) or None
    """
    # Normalize/flatten
    feat = np.array(p_t, dtype=np.float32)

    # Last action one-hot (9 dim)
    act_oh = np.zeros(9, dtype=np.float32)
    if last_a is not None:
        act_oh[last_a] = 1.0

    # Last observation one-hot: for each tiger (3 values: 0,1,2) -> 6 dims
    obs_oh = np.zeros(6, dtype=np.float32)
    if last_o is not None:
        o1, o2 = last_o
        obs_oh[o1] = 1.0
        obs_oh[3 + o2] = 1.0

    return np.concatenate([feat, act_oh, obs_oh])


def check_gpu():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def main():
    env = TwoTigersEnv(max_episode_steps=1000)
    action_space = env.action_space_()
    state_space = env.state_space_()  # 4 (beliefs)

    # PSR predictor input dim = belief(4) + action_oh(9) + obs_oh(6) = 19
    psr_input_dim = 19
    psr = PSRPredictor(psr_input_dim)
    psr_buffer = PSRBuffer(psr_input_dim, size=5000)
    psr_optimizer = optim.Adam(psr.parameters(), lr=1e-3)
    psr_loss_fn = nn.BCELoss()

    # DQN will take as input: belief(4) + psr_preds(2) = 6 dims
    enhanced_state_dim = 4 + 2

    device = check_gpu()
    psr.to(device)

    # DQN agent
    dqn = DQNAgent(enhanced_state_dim, action_space, device=device, lr=1e-3, gamma=0.99, batch_size=64)

    writer = SummaryWriter()

    eps_start = 0.5
    eps_end = 0.01
    eps_decay = 0.995

    episodes = 1000  # shorter for quick experiments; change as needed

    for episode in trange(episodes):
        epsilon = max(eps_end, eps_start * (eps_decay ** episode))
        obs, info = env.reset()

        # initial belief
        p_t = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        last_a = None
        last_o = None

        total_reward = 0
        steps = 0
        losses = []

        while True:
            # compute PSR features (predictions) using current inputs
            psr_in = build_input_features(p_t, last_a, last_o)
            with torch.no_grad():
                psr_tensor = torch.FloatTensor(psr_in).unsqueeze(0).to(device)
                psr_pred = psr(psr_tensor).cpu().numpy().reshape(-1)  # 2 dims

            # enhanced state
            state_enh = np.concatenate([p_t, psr_pred]).astype(np.float32)

            # select action
            action_int = dqn.q_net.sample_action(torch.FloatTensor(state_enh).to(device), epsilon=epsilon)

            next_obs, reward, done, truncated, info = env.step(action_int)
            episode_done = done or truncated

            # update PSR training buffer if the action taken was a listen on either tiger
            act1 = action_int // 3
            act2 = action_int % 3

            # For PSR training we only have supervised signal when we listen.
            # If act1==0 (listen on tiger1), target1 = 1 if obs1==1 (OL), else 0
            # If not listening, we still store a sample with no-signal? Here we store only when listen.
            if act1 == 0 or act2 == 0:
                # targets: probability of OL for each tiger (1 if observed OL, 0 otherwise)
                t1 = 1.0 if (next_obs[0] == 1) else 0.0
                t2 = 1.0 if (next_obs[1] == 1) else 0.0
                psr_buffer.add(psr_in, np.array([t1, t2], dtype=np.float32))

            # update belief state using oracle bayes update (same as original script)
            p_t_tiger1 = p_t[0:2]
            p_t_tiger2 = p_t[2:4]
            a_tuple = ("AL", "AR", "AE")[act1], ("AL", "AR", "AE")[act2]
            # map observation ints to strings: 0->"None", 1->"OL", 2->"OR"
            obs_map = {0: None, 1: "OL", 2: "OR"}
            o_tuple = (obs_map[next_obs[0]], obs_map[next_obs[1]])

            p_t_next_t1 = None
            p_t_next_t2 = None
            # Update per-tiger
            # For listen: use Bayesian update; for open: reset to uniform
            if a_tuple[0] == "AE":
                if o_tuple[0] == "OL":
                    numL = 0.85 * p_t_tiger1[0]
                    numR = 0.15 * p_t_tiger1[1]
                else:
                    numL = 0.15 * p_t_tiger1[0]
                    numR = 0.85 * p_t_tiger1[1]
                tot = numL + numR
                p_t_next_t1 = np.array([numL / tot, numR / tot])
            else:
                p_t_next_t1 = np.array([0.5, 0.5])

            if a_tuple[1] == "AE":
                if o_tuple[1] == "OL":
                    numL = 0.85 * p_t_tiger2[0]
                    numR = 0.15 * p_t_tiger2[1]
                else:
                    numL = 0.15 * p_t_tiger2[0]
                    numR = 0.85 * p_t_tiger2[1]
                tot = numL + numR
                p_t_next_t2 = np.array([numL / tot, numR / tot])
            else:
                p_t_next_t2 = np.array([0.5, 0.5])

            p_t_next = np.concatenate([p_t_next_t1, p_t_next_t2])

            # push transition to DQN buffer (use enhanced states)
            dqn.replay_buffer.put(state_enh, action_int, reward, np.concatenate([p_t_next, psr_pred]), episode_done)

            # train DQN
            if len(dqn.replay_buffer) >= dqn.batch_size:
                loss = dqn.update()
                if loss is not None:
                    losses.append(loss)
                if steps % 5 == 0:
                    dqn.soft_update(tau=1e-2)

            # train PSR predictor from its buffer occasionally
            if psr_buffer.size >= 32:
                psr_inputs, psr_targets = psr_buffer.sample(batch=32)
                psr_inputs_t = torch.FloatTensor(psr_inputs).to(device)
                psr_targets_t = torch.FloatTensor(psr_targets).to(device)
                preds = psr(psr_inputs_t)
                loss_psr = psr_loss_fn(preds, psr_targets_t)
                psr_optimizer.zero_grad()
                loss_psr.backward()
                psr_optimizer.step()

            total_reward += reward
            steps += 1
            last_a = action_int
            last_o = (int(next_obs[0]), int(next_obs[1]))
            p_t = p_t_next

            if episode_done:
                break

        avg_loss = np.mean(losses) if len(losses) > 0 else 0.0
        print(f"Ep {episode+1} Done | Reward: {total_reward} | Steps: {steps} | DQN loss: {avg_loss:.4f}")

        writer.add_scalar("Reward/episode/psr_dqn", total_reward, episode)
        writer.add_scalar("Loss/episode/psr_dqn", avg_loss, episode)

    writer.close()


if __name__ == '__main__':
    main()
