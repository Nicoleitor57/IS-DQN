import random
from typing import Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class PrioritizedReplayBuffer:
    """
    Buffer de repetición priorizado (PER) basado en numpy.
    No es tan eficiente como un SumTree, pero es mucho más simple de leer.
    """
    def __init__(self, obs_dim: int, size: int, batch_size: int = 32, alpha: float = 0.6):
        # Buffers de datos (igual que antes)
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size], dtype=np.float32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        
        # --- NUEVO para PER ---
        self.priorities = np.zeros(size, dtype=np.float32) # Almacena prioridades
        self.epsilon = 1e-6  # Para evitar prioridades de 0
        self.alpha = alpha   # Hiperparámetro (0=uniforme, 1=totalmente priorizado)
        # ---------------------
        
        self.max_size = size
        self.batch_size = batch_size
        self.ptr = 0
        self.size = 0

    def put(self, obs, act, rew, next_obs, done):
        # --- NUEVO para PER ---
        # Asignar la prioridad más alta vista hasta ahora a las nuevas transiciones
        # para asegurar que se muestreen al menos una vez.
        max_prio = np.max(self.priorities) if self.size > 0 else 1.0
        self.priorities[self.ptr] = max_prio
        # ---------------------

        # Almacenar la transición (igual que antes)
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, beta: float = 0.4) -> Dict[str, np.ndarray]:
        """
        Muestrea del buffer usando prioridades y calcula los pesos 
        de "Importance Sampling (IS)".
        """
        if self.size == 0:
            return None, None, None

        # 1. Obtener probabilidades priorizadas
        prios = self.priorities[:self.size]
        probs = prios ** self.alpha
        probs /= probs.sum()

        # 2. Muestrear índices basados en probabilidades
        idxs = np.random.choice(self.size, self.batch_size, p=probs, replace=True)

        # 3. Obtener las muestras de datos
        samples = dict(
            obs=self.obs_buf[idxs],
            next_obs=self.next_obs_buf[idxs],
            acts=self.acts_buf[idxs],
            rews=self.rews_buf[idxs],
            done=self.done_buf[idxs]
        )

        # --- NUEVO para PER: Pesos de Importance Sampling ---
        # 4. Calcular pesos IS (w_j = (N * P(j))^-beta)
        total = self.size
        weights = (total * probs[idxs]) ** (-beta)
        weights /= weights.max() # Normalizar para estabilidad
        weights = np.array(weights, dtype=np.float32)
        # --------------------------------------------------

        return samples, idxs, weights

    def update_priorities(self, batch_indices, batch_priorities):
        """Actualiza las prioridades de las transiciones muestreadas."""
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = (prio + self.epsilon)

    def __len__(self):
        return self.size
    
    
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
    
    
    
class DQNAgent:
    def __init__(self, 
                 state_space: int, 
                 action_space: int, 
                 device='cuda', 
                 lr=1e-3, 
                 gamma=0.99, 
                 batch_size=64,
                 buffer_size=100_000,
                 alpha=0.6,          # <-- NUEVO: Hiperparámetro PER
                 beta_start=0.4,     # <-- NUEVO: Beta inicial para IS
                 beta_frames=100_000 # <-- NUEVO: Pasos para llegar a beta=1.0
                ):
        
        self.device = device
        self.gamma = gamma
        self.batch_size = batch_size

        # Q-networks
        self.q_net = QNet(state_space, action_space).to(device)
        self.target_q_net = QNet(state_space, action_space).to(device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        # --- CAMBIO: Usar el nuevo buffer ---
        self.replay_buffer = PrioritizedReplayBuffer(
            obs_dim=state_space, 
            size=buffer_size, 
            batch_size=batch_size,
            alpha=alpha
        )
        
        # --- NUEVO: Hiperparámetros PER ---
        self.beta_start = beta_start
        self.beta = beta_start
        self.beta_increment = (1.0 - beta_start) / beta_frames
        self.total_steps = 0 # Para seguir el annealing de beta
        
    def store_transition(self, obs, act, rew, next_obs, done):
        """Método de ayuda para poblar el buffer"""
        self.replay_buffer.put(obs, act, rew, next_obs, done)

    def update(self):
        """Entrena Q-network desde el PER. Devuelve la pérdida."""
        if len(self.replay_buffer) < self.batch_size:
            return None  # No hay suficientes muestras
        
        # --- CAMBIO 1: Anneal beta y muestrear del PER ---
        self.beta = min(1.0, self.beta_start + self.total_steps * self.beta_increment)
        self.total_steps += 1
        
        samples, idxs, weights = self.replay_buffer.sample(self.beta)
        
        # ------------------------------------------------
        
        states = torch.FloatTensor(samples['obs']).to(self.device)
        actions = torch.LongTensor(samples['acts'].reshape(-1,1)).to(self.device)
        rewards = torch.FloatTensor(samples['rews'].reshape(-1,1)).to(self.device)
        next_states = torch.FloatTensor(samples['next_obs']).to(self.device)
        dones = torch.FloatTensor(samples['done'].reshape(-1,1)).to(self.device)
        
        # --- CAMBIO 2: Convertir pesos IS a Tensor ---
        weights_tensor = torch.FloatTensor(weights).reshape(-1, 1).to(self.device)
        # ---------------------------------------------

        # Compute target
        q_target_max = self.target_q_net(next_states).max(1)[0].unsqueeze(1).detach()
        targets = rewards + self.gamma * q_target_max * (1 - dones)

        q_out = self.q_net(states)
        q_a = q_out.gather(1, actions)

        # --- CAMBIO 3: Calcular pérdida ponderada y errores TD ---
        # Usamos reduction='none' para obtener la pérdida por elemento
        elementwise_loss = F.smooth_l1_loss(q_a, targets, reduction='none')
        
        # Ponderar la pérdida por los pesos IS
        loss = (elementwise_loss * weights_tensor).mean()
        
        # Calcular los errores TD (necesarios para actualizar prioridades)
        # .detach() es crucial para no propagar el gradiente de la prioridad
        td_errors = (targets - q_a).abs().detach().cpu().numpy().squeeze()
        # -------------------------------------------------------

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # --- CAMBIO 4: Actualizar las prioridades en el buffer ---
        self.replay_buffer.update_priorities(idxs, td_errors)
        # ------------------------------------------------------
        
        return loss.item()

    def soft_update(self, tau=1e-2):
        """Soft update target network"""
        for target_param, local_param in zip(self.target_q_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0 - tau)*target_param.data)

    def save_model(self, path='dqn_model.pth'):
        torch.save(self.q_net.state_dict(), path)

    def load_model(self, path):
        self.q_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_q_net.load_state_dict(self.q_net.state_dict())