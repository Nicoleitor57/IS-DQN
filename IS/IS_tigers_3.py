"""
IS_tigers_3.py

Spectral PSR (Predictive State Representation) + DQN variant.

Theory (brief):
- Histories H: recent (action, obs) sequences (here: one-step).
- Tests T: outcomes of future actions. We define tests as: "if I listen (AE) 
  next on tiger i, will I hear OL (1) or OR (0)?" → 2 tests per tiger.
- Empirical data matrix P_HT: |H| x |T|, where P_HT[h,t] = count(obs=test_outcome | h).
- SVD of P_HT: U Σ V^T. Low-rank approx: keep r singular vectors.
- PSR state: projection of history into r-dim space via U_r (top r cols of U).
- At runtime: map current history to embedding, concatenate with belief for DQN.

This is a lightweight approximation of spectral PSR methods from literature
(e.g., Boots et al. 2013, Holmes et al. 2010).
"""

from Entornos.TwoTigersEnv import TwoTigersEnv
from .DQN import DQNAgent

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict


class SpectralPSR:
    """
    Learn a low-rank PSR via SVD of empirical history-test matrix.
    """
    def __init__(self, rank=2, max_histories=1000):
        self.rank = rank
        self.max_histories = max_histories
        
        # storage for empirical counts: history_key -> {test_key: count}
        self.history_test_counts = defaultdict(lambda: defaultdict(int))
        self.history_list = []
        self.U_r = None  # learned embedding matrix (will be set after SVD)
        self.history_to_idx = {}  # map history tuples to rows in matrix
        
    def add_observation(self, history_tuple, test_outcome):
        """
        history_tuple: (action_last, obs_last) or similar discrete encoding.
        test_outcome: tuple (test1_result, test2_result) where each is 0 or 1.
        """
        self.history_test_counts[history_tuple]['test_tiger1'] += test_outcome[0]
        self.history_test_counts[history_tuple]['test_tiger2'] += test_outcome[1]
        
        # track count of this test outcome for normalization later
        self.history_test_counts[history_tuple]['_count'] += 1
        
        if history_tuple not in self.history_to_idx:
            if len(self.history_list) < self.max_histories:
                self.history_to_idx[history_tuple] = len(self.history_list)
                self.history_list.append(history_tuple)
    
    def fit(self):
        """
        Compute SVD of empirical P_HT matrix and extract top-r singular vectors.
        Returns embedding dimension or 0 if insufficient data.
        """
        if len(self.history_list) < 2:
            print("⚠️  SpectralPSR: not enough histories to fit (need >= 2).")
            return 0
        
        n_hist = len(self.history_list)
        n_tests = 2  # test per tiger
        
        P_HT = np.zeros((n_hist, n_tests), dtype=np.float32)
        
        for h_idx, h in enumerate(self.history_list):
            counts = self.history_test_counts[h]
            total = counts.get('_count', 1)
            P_HT[h_idx, 0] = counts.get('test_tiger1', 0) / max(total, 1)
            P_HT[h_idx, 1] = counts.get('test_tiger2', 0) / max(total, 1)
        
        # SVD
        try:
            U, S, Vt = np.linalg.svd(P_HT, full_matrices=False)
        except Exception as e:
            print(f"⚠️  SVD failed: {e}")
            return 0
        
        # keep top-r
        r_eff = min(self.rank, len(S))
        self.U_r = U[:, :r_eff]  # n_hist x r_eff
        
        print(f"✅ SpectralPSR fitted: {n_hist} histories, rank {r_eff}, explained variance {S[:r_eff].sum() / S.sum():.2%}")
        
        return r_eff
    
    def embed_history(self, history_tuple):
        """
        Map a history to its embedding. If unknown, return zeros.
        """
        if self.U_r is None:
            return np.zeros(self.rank, dtype=np.float32)
        
        if history_tuple in self.history_to_idx:
            idx = self.history_to_idx[history_tuple]
            return self.U_r[idx].astype(np.float32)
        else:
            # unknown history: zero embedding
            return np.zeros(self.U_r.shape[1], dtype=np.float32)


def check_gpu():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")
    return device


def main(episodes=200, warmup_transitions=1000, rank=2):
    """
    Main training loop with spectral PSR.
    
    Phase 1: Collect warmup data with random policy, fit PSR.
    Phase 2: Train DQN using PSR-embedded features.
    """
    
    env = TwoTigersEnv(max_episode_steps=1000)
    action_space = env.action_space_()
    device = check_gpu()
    
    # Initialize Spectral PSR
    psr = SpectralPSR(rank=rank, max_histories=2000)
    
    print(f"\n{'='*50}")
    print(f"Phase 1: Warmup - collecting {warmup_transitions} transitions...")
    print(f"{'='*50}\n")
    
    # Warmup phase: random policy
    warmup_steps = 0
    while warmup_steps < warmup_transitions:
        obs, info = env.reset()
        p_t = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        last_a = None
        last_o = None
        
        while warmup_steps < warmup_transitions:
            # encode history
            history_key = (last_a, last_o) if last_a is not None else (None, None)
            
            # random action
            action_int = np.random.randint(0, action_space)
            next_obs, reward, done, truncated, info = env.step(action_int)
            episode_done = done or truncated
            
            # if action includes listen, record test outcome
            act1 = action_int // 3
            act2 = action_int % 3
            
            # map obs to test outcome: 1 if OL (obs==1), 0 otherwise
            test_result = (
                1 if next_obs[0] == 1 else 0,  # tiger1: OL = 1, OR = 0
                1 if next_obs[1] == 1 else 0   # tiger2: OL = 1, OR = 0
            )
            psr.add_observation(history_key, test_result)
            
            last_a = action_int
            last_o = (int(next_obs[0]), int(next_obs[1]))
            warmup_steps += 1
            
            if episode_done:
                break
    
    # Fit PSR
    psr_rank = psr.fit()
    if psr_rank == 0:
        print("⚠️  PSR fitting failed; using zero embeddings.")
        psr_rank = rank
    
    # DQN input: belief (4) + PSR embedding (psr_rank)
    dqn_input_dim = 4 + psr_rank
    dqn = DQNAgent(dqn_input_dim, action_space, device=device, lr=1e-2, gamma=0.99, batch_size=64)
    writer = SummaryWriter()
    
    print(f"\n{'='*50}")
    print(f"Phase 2: Training DQN with PSR features ({dqn_input_dim}-dim input)...")
    print(f"{'='*50}\n")
    
    eps_start = 0.5
    eps_end = 0.01
    eps_decay = 0.995
    
    for episode in trange(episodes):
        epsilon = max(eps_end, eps_start * (eps_decay ** episode))
        obs, info = env.reset()
        p_t = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        last_a = None
        last_o = None
        
        total_reward = 0
        steps = 0
        losses = []
        
        while True:
            # get PSR embedding
            history_key = (last_a, last_o) if last_a is not None else (None, None)
            psr_emb = psr.embed_history(history_key)
            state_enh = np.concatenate([p_t, psr_emb]).astype(np.float32)
            
            # DQN action
            action_int = dqn.q_net.sample_action(torch.FloatTensor(state_enh).to(device), epsilon=epsilon)
            
            next_obs, reward, done, truncated, info = env.step(action_int)
            episode_done = done or truncated
            
            # decode action
            act1 = action_int // 3
            act2 = action_int % 3
            a_tuple = ("AL", "AR", "AE")[act1], ("AL", "AR", "AE")[act2]
            obs_map = {0: None, 1: "OL", 2: "OR"}
            o_tuple = (obs_map[next_obs[0]], obs_map[next_obs[1]])
            
            # update beliefs (oracle Bayes)
            p_t_tiger1 = p_t[0:2]
            p_t_tiger2 = p_t[2:4]
            
            if a_tuple[0] == "AE":
                if o_tuple[0] == "OL":
                    numL = 0.85 * p_t_tiger1[0]
                    numR = 0.15 * p_t_tiger1[1]
                else:
                    numL = 0.15 * p_t_tiger1[0]
                    numR = 0.85 * p_t_tiger1[1]
                tot = numL + numR
                p_t_next_tiger1 = np.array([numL / tot, numR / tot])
            else:
                p_t_next_tiger1 = np.array([0.5, 0.5])
            
            if a_tuple[1] == "AE":
                if o_tuple[1] == "OL":
                    numL = 0.85 * p_t_tiger2[0]
                    numR = 0.15 * p_t_tiger2[1]
                else:
                    numL = 0.15 * p_t_tiger2[0]
                    numR = 0.85 * p_t_tiger2[1]
                tot = numL + numR
                p_t_next_tiger2 = np.array([numL / tot, numR / tot])
            else:
                p_t_next_tiger2 = np.array([0.5, 0.5])
            
            p_t_next = np.concatenate([p_t_next_tiger1, p_t_next_tiger2])
            
            # next state with PSR
            next_history_key = (action_int, (int(next_obs[0]), int(next_obs[1])))
            next_psr_emb = psr.embed_history(next_history_key)
            next_state_enh = np.concatenate([p_t_next, next_psr_emb]).astype(np.float32)
            
            # store in replay buffer
            dqn.replay_buffer.put(state_enh, action_int, reward, next_state_enh, episode_done)
            
            # train DQN
            if len(dqn.replay_buffer) >= dqn.batch_size:
                loss = dqn.update()
                if loss is not None:
                    losses.append(loss)
                if steps % 5 == 0:
                    dqn.soft_update(tau=1e-2)
            
            total_reward += reward
            steps += 1
            last_a = action_int
            last_o = (int(next_obs[0]), int(next_obs[1]))
            p_t = p_t_next
            
            if episode_done:
                break
        
        avg_loss = np.mean(losses) if len(losses) > 0 else 0.0
        print(f"Ep {episode+1} Done | Reward: {total_reward} | Steps: {steps} | Loss: {avg_loss:.4f}")
        
        writer.add_scalar("Reward/episode/spectral_psr", total_reward, episode)
        writer.add_scalar("Loss/episode/spectral_psr", avg_loss, episode)
    
    writer.close()


if __name__ == '__main__':
    main(episodes=1000, warmup_transitions=1000, rank=4)
