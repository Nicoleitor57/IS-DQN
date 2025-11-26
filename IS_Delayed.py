
# import gymnasium as gym
# import numpy as np
# import pandas as pd
# import time
# import os
# import glob
# import matplotlib.pyplot as plt
# import random
# from collections import deque
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim


# try:
#     from Entornos.StocasticDelayedObsEnv import StochasticGridEnv
# except ImportError:
#     print("ERROR CRÍTICO: No se encuentra 'Entornos/KeyDoorMazeEnv.py'.")
#     exit()



# # ==============================================================================
# # 1. WRAPPER: BELIEF STATE (PSR) + ENTROPY
# # ==============================================================================


# class DelayedBeliefWrapper(gym.Wrapper):
#     """
#     Wrapper que implementa el filtro de Anclaje-Propagación para DelayedObsEnv.
    
#     Salida (Observación): [b(s1), b(s2), ..., b(s|S|), Entropía(b)]
#     """

#     def __init__(self, env):
#         super().__init__(env)
#         self.base_env = env.unwrapped # Acceso al entorno base (StochasticGridEnv)
#         self.H = self.base_env.size  # Asumimos N=10
#         self.W = self.base_env.size
#         self.slip_prob = self.base_env.slip_prob # 0.1
        
#         # 2 estados de llave: 0=Falso, 1=Verdadero
#         self.num_key_states = 2
#         self.num_states = self.H * self.W * self.num_key_states # |S| = 200
        
#         # 1. Almacenar el historial (se accederá al deque del base_env)
#         self.delay_steps = self.base_env.delay_steps
        
#         # 2. Pre-calcular la matriz de Transición T(s'|s, a)
#         self.T_matrix = self._build_transition_matrix() # Forma: |S| x |A| x |S|
        
#         # 3. Definir espacio de observación (Belief + Entropía)
#         self.observation_space = gym.spaces.Box(
#             low=0.0, high=20.0, # Límite superior holgado
#             shape=(self.num_states + 1,), 
#             dtype=np.float32
#         )
#         # Inicializar belief
#         self.current_belief = np.zeros(self.num_states, dtype=np.float32)


#     # --- Utilitarios de Índices ---
    
#     def _state_to_idx(self, x, y, k):
#         # x, y son coordenadas. k es 0 o 1.
#         if 0 <= x < self.H and 0 <= y < self.W:
#             return k * (self.H * self.W) + x * self.W + y
#         return -1 # Fuera de límites

#     def _idx_to_state(self, idx):
#         area = self.H * self.W
#         k = idx // area
#         rem = idx % area
#         x = rem // self.W
#         y = rem % self.W
#         return x, y, k

#     # --- Constructor del Modelo Interno (T) ---
    
#     def _get_transition_prob(self, current_x, current_y, current_k, action, next_x, next_y, next_k):
#         """
#         Calcula T(s'|s,a) = P(x', y', k' | x, y, k, a). 
#         Esta función debe coincidir con la lógica estocástica del base_env.
#         """
#         # 1. Transición de Posición (Estocasticidad)
#         _delta = [(-1, 0), (1, 0), (0, -1), (0, 1)]
#         delta = _delta[action]
        
#         prob = 0.0
        
#         # A) Escenario de Éxito (1 - epsilon)
#         target_x, target_y = current_x + delta[0], current_y + delta[1]
#         if (next_x, next_y) == (target_x, target_y):
#             prob += (1.0 - self.slip_prob)
            
#         # B) Escenario de Fallo (epsilon) - El agente se queda quieto
#         elif (next_x, next_y) == (current_x, current_y):
#             prob += self.slip_prob
            
#         # C) Redirigir la probabilidad si el movimiento es a un Muro
#         is_wall = self.base_env._grid[target_x, target_y] == 1
#         is_locked_door = (target_x, target_y) == self.base_env._door_location and not current_k # Si no tiene llave
        
#         if (is_wall or is_locked_door):
#             # Si el movimiento exitoso choca con un obstáculo, se queda en (x,y).
#             if (next_x, next_y) == (current_x, current_y):
#                  prob += (1.0 - self.slip_prob) # La probabilidad de éxito se redirige al estado actual
#             prob = prob * 0.0 # Ajuste final para evitar doble conteo y asegurar que solo se cuentan las transiciones válidas.
            
#             # Nota: La lógica del env real maneja esto devolviendo la ubicación actual.
#             # Aquí, solo nos importa el 1-epsilon / epsilon split.

#         # La estocasticidad del env es solo quedarse quieto (prob epsilon).
#         # Lo más simple y seguro es calcular la probabilidad de moverse a (nx, ny) y luego aplicar las reglas del entorno.

#         # --- Lógica simplificada de T(s'|s, a) ---
#         target_pos = (current_x + delta[0], current_y + delta[1])
        
#         # 1. Calcular la Posición FINAL si el movimiento FUE exitoso
#         success_x, success_y = target_pos
#         if self.base_env._grid[success_x, success_y] == 1 or ((success_x, success_y) == self.base_env._door_location and not current_k):
#             success_x, success_y = current_x, current_y # Colisión: se queda quieto
            
#         # 2. Calcular la Posición FINAL si el movimiento FALLÓ (se quedó quieto)
#         failure_x, failure_y = current_x, current_y
#         # Si quedarse quieto está en un muro (no debería pasar), se queda quieto.
#         # No hay lógica de key pickup para quedarse quieto.

#         T_prob = 0.0
#         if (next_x, next_y) == (success_x, success_y):
#             T_prob += (1.0 - self.slip_prob)
#         if (next_x, next_y) == (failure_x, failure_y):
#             T_prob += self.slip_prob
        
#         # 3. Transición de Llave (Determinista dado el movimiento)
#         next_k_target = current_k
#         if next_k == 1 and current_k == 0 and (next_x, next_y) == self.base_env._key_location:
#             next_k_target = 1 # Solo se puede recoger si no tenía llave.
            
#         if next_k_target != next_k:
#             return 0.0 # Transición de llave incorrecta
            
#         return T_prob

#     def _build_transition_matrix(self):
#         """Pre-calcula la matriz T(|S| x |A| x |S|)"""
#         T = np.zeros((self.num_states, 4, self.num_states), dtype=np.float32)
        
#         for s_idx in range(self.num_states):
#             current_x, current_y, current_k = self._idx_to_state(s_idx)
            
#             # Solo consideramos estados no-muro
#             if self.base_env._grid[current_x, current_y] == 1: 
#                 continue
                
#             for a_idx in range(4):
#                 for s_prime_idx in range(self.num_states):
#                     next_x, next_y, next_k = self._idx_to_state(s_prime_idx)
                    
#                     prob = self._get_transition_prob(
#                         current_x, current_y, current_k, 
#                         a_idx, 
#                         next_x, next_y, next_k
#                     )
                    
#                     if prob > 1e-9:
#                         T[s_idx, a_idx, s_prime_idx] = prob
                        
#         return T

#     # --- Lógica del Rollout (Propagación) ---

#     def _rollout_belief(self, s_past_idx, action_history):
#         """
#         Implementa la Propagación (Rollout) del belief: 
#         Aplica T_matrix k veces sobre el one-hot inicial (Anclaje).
#         """
#         # 1. Anclaje (one-hot en s_past)
#         belief = np.zeros(self.num_states, dtype=np.float32)
#         belief[s_past_idx] = 1.0
        
#         # 2. Propagación (k veces)
#         for t_step, a in enumerate(action_history):
#             # Multiplicación matriz-vector: T(s'|s, a) * b(s)
#             # T[s, a, s'] => forma (S, S) para una acción fija a
#             T_a = self.T_matrix[:, a, :] # T[s, a, s']
            
#             # b_next[s'] = sum_s T[s, a, s'] * b[s]
#             # Esto es lo mismo que: b_next = b @ T_a (si T_a fuera (S, S'))
#             # Para la forma (S, S) que tenemos, usamos T_a.T
#             # b_next[s'] = sum_s T[s, a, s'] * b[s]
            
#             # La transpuesta debe ser para T(s'|s, a)
#             # T_a es T(s, s' | a) - debemos transponer para T(s'|s, a)
#             # Simplemente usamos np.dot con la forma correcta.
            
#             # Nueva creencia = creencia_anterior @ T_a
#             belief = belief @ T_a 
            
#         return belief

#     def _calculate_entropy(self, belief):
#         b = np.clip(belief, 1e-9, 1.0)
#         max_entropy = np.log2(self.num_states)
#         entropy = -np.sum(b * np.log2(b)) / max_entropy
#         return np.float32(entropy)

#     # --- Métodos Gym ---

#     def reset(self, **kwargs):
#         # El reset del base_env ya inicializa el state_history deque.
#         obs, info = self.env.reset(**kwargs)
        
#         # obs (del DelayedObsEnv) es el estado inicial one-hot [x, y, k]
#         # Obtenemos el índice del estado inicial
#         x, y, k = obs
#         s_past_idx = self._state_to_idx(x, y, k)
        
#         # El belief inicial es one-hot en ese s_past_idx
#         self.current_belief = np.zeros(self.num_states, dtype=np.float32)
#         self.current_belief[s_past_idx] = 1.0
        
#         # La entropía al inicio es 0 (certeza total)
#         h = self._calculate_entropy(self.current_belief)
        
#         return np.concatenate([self.current_belief, [h]]), info

#     def step(self, action):
#         # 1. Paso en el entorno base (obtiene obs_t = s_{t-k} y actualiza action_history)
#         obs_past_state, reward, terminated, truncated, info = self.env.step(action)
        
#         # 2. El Anclaje: Convertir la observación (el estado pasado) a índice one-hot
#         x_past, y_past, k_past = obs_past_state
#         s_past_idx = self._state_to_idx(x_past, y_past, k_past)
        
#         # 3. Obtener el historial de acciones
#         action_history = self.env.unwrapped.action_history
        
#         # 4. Rollout (Propagación)
#         self.current_belief = self._rollout_belief(s_past_idx, action_history)
        
#         # 5. Salida
#         h = self._calculate_entropy(self.current_belief)
#         state = np.concatenate([self.current_belief, [h]])
        
#         return state, reward, terminated, truncated, info


import gymnasium as gym
import numpy as np
import pandas as pd
import time
import os
import glob
import matplotlib.pyplot as plt
import random
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys

# --- IMPORTACIÓN DE ENTORNOS (Asumimos que están definidos correctamente) ---
try:
    from Entornos.StochasticDelayedObsEnv import DelayedStochasticObsEnv
    # Si DelayedStochasticObsEnv está en otro archivo, ajusta la importación.
    # Por ahora, lo definiremos aquí para que el script sea autocontenido.
except ImportError:
    print("Error: Asegúrate de que StochasticGridEnv.py esté definido.")
    sys.exit(1)

# ==============================================================================
# CLASE DelayedStochasticObsEnv (Asumimos el código que proporcionaste)
# ... (Tu código para DelayedStochasticObsEnv va aquí) ...

# ------------------------------------------------------------------------------
# CLASE Belief Wrapper (Implementa el Filtro de Propagación)
# ------------------------------------------------------------------------------
class DelayedBeliefWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        
        # Acceso a los atributos de la base_env (StochasticGridEnv)
        self.base_env = env.base_env # Acceso a la dinámica ruidosa
        self.H = self.base_env.size
        self.W = self.base_env.size
        self.slip_prob = self.base_env.slip_prob
        self.delay_steps = self.env.delay_steps # Delay (k)
        
        # |S| = N^2 * 2 (Posición x Llave)
        self.num_key_states = 2 # 0=Falso, 1=Verdadero
        self.num_states = self.H * self.W * self.num_key_states # 200
        
        self.T_matrix = self._build_transition_matrix() # Forma: |S| x |A| x |S|
        
        # Espacio: Belief (200) + Entropía (1)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=20.0, shape=(self.num_states + 1,), dtype=np.float32
        )
        self.current_belief = np.zeros(self.num_states, dtype=np.float32)

    # --- Utilitarios de Índices ---
    # def _state_to_idx(self, x, y, k):
    #     if 0 <= x < self.H and 0 <= y < self.W:
    #         return k * (self.H * self.W) + x * self.W + y
    #     return -1
    def _state_to_idx(self, x, y, k):
        # Conversión explícita a int para evitar el warning de desbordamiento de NumPy
        H, W = int(self.H), int(self.W)
        k, x, y = int(k), int(x), int(y)

        if 0 <= x < H and 0 <= y < W:
            return k * (H * W) + x * W + y

        return -1

    
    def _idx_to_state(self, idx):
        area = self.H * self.W
        k = idx // area
        rem = idx % area
        x = rem // self.W
        y = rem % self.W
        return x, y, k

    # --- Constructor de T ---
    def _get_transition_prob(self, s, a, s_prime):
        """Calcula T(s'|s,a) = P(x', y', k' | x, y, k, a)."""
        current_x, current_y, current_k = s
        next_x, next_y, next_k = s_prime
        
        _delta = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        delta = _delta[a]
        T_prob = 0.0
        
        # 1. Calcular la Posición FINAL si el movimiento fue un ÉXITO (1 - slip_prob)
        success_x_t, success_y_t = current_x + delta[0], current_y + delta[1]
        
        # --- VERIFICACIÓN DE LÍMITES Y COLISIÓN ---
        is_out_of_bounds = not (0 <= success_x_t < self.H and 0 <= success_y_t < self.W)
        
        if not is_out_of_bounds:
            # Ahora es seguro acceder a la cuadrícula
            is_wall = self.base_env._grid[success_x_t, success_y_t] == 1
            is_locked_door = (success_x_t, success_y_t) == self.base_env._door_location and not current_k
        else:
            is_wall = False
            is_locked_door = False

        if is_out_of_bounds or is_wall or is_locked_door:
            success_x, success_y = current_x, current_y # Se queda quieto
        else:
            success_x, success_y = success_x_t, success_y_t # Se mueve
            
        # 2. Posición final si el movimiento fue un FALLO (slip_prob)
        failure_x, failure_y = current_x, current_y # El fallo es quedarse quieto

        # 3. Aplicar las probabilidades de Éxito/Fallo
        if (next_x, next_y) == (success_x, success_y):
            T_prob += (1.0 - self.slip_prob)
        if (next_x, next_y) == (failure_x, failure_y):
            T_prob += self.slip_prob
        
        # 4. Transición de Llave (Determinista dado el movimiento)
        next_k_target = current_k
        if next_k == 1 and current_k == 0 and (next_x, next_y) == self.base_env._key_location:
            next_k_target = 1
            
        if next_k_target != next_k:
            return 0.0
            
        return T_prob
    
    def _build_transition_matrix(self):
        """Pre-calcula la matriz T(|S| x |A| x |S|)"""
        T = np.zeros((self.num_states, 4, self.num_states), dtype=np.float32)
        
        for s_idx in range(self.num_states):
            s = self._idx_to_state(s_idx)
            # Solo consideramos estados no-muro
            if self.base_env._grid[s[0], s[1]] == 1: continue 
            
            for a_idx in range(4):
                for s_prime_idx in range(self.num_states):
                    s_prime = self._idx_to_state(s_prime_idx)
                    prob = self._get_transition_prob(s, a_idx, s_prime)
                    if prob > 1e-9:
                        T[s_idx, a_idx, s_prime_idx] = prob
        return T
    
    
    # --- Lógica del Rollout (Propagación) ---
    def _rollout_belief(self, s_past_idx, action_history):
        """Aplica la Propagación (Rollout) del belief k veces."""
        # 1. Anclaje (one-hot en s_past)
        belief = np.zeros(self.num_states, dtype=np.float32)
        belief[s_past_idx] = 1.0
        
        # 2. Propagación (k veces)
        for a in action_history:
            T_a = self.T_matrix[:, a, :] # T[s, a, s']
            belief = belief @ T_a 
            
        return belief

    def _calculate_entropy(self, belief):
        b = np.clip(belief, 1e-9, 1.0)
        max_entropy = np.log2(self.num_states)
        entropy = -np.sum(b * np.log2(b)) / max_entropy
        return np.float32(entropy)

    # --- Métodos Gym ---
    def reset(self, **kwargs):
        obs_past_state, info = self.env.reset(**kwargs)
        
        # Anclaje: s_past es la primera observación [x, y, k]
        x, y, k = obs_past_state
        s_past_idx = self._state_to_idx(x, y, k)
        
        self.current_belief = np.zeros(self.num_states, dtype=np.float32)
        self.current_belief[s_past_idx] = 1.0
        
        h = self._calculate_entropy(self.current_belief)
        
        # La observación es el belief inicial + entropía
        return np.concatenate([self.current_belief, [h]]), info

    def step(self, action):
        # 1. Paso en el entorno base (obtiene obs_t = s_{t-k} y actualiza action_history)
        obs_past_state, reward, terminated, truncated, info = self.env.step(action)
        
        # 2. Anclaje: Convertir la observación (s_{t-k}) a índice
        x_past, y_past, k_past = obs_past_state
        s_past_idx = self._state_to_idx(x_past, y_past, k_past)
        
        # 3. Rollout
        action_history = self.env.action_history # Accede al historial del DelayedObsEnv
        self.current_belief = self._rollout_belief(s_past_idx, action_history)
        
        # 4. Salida
        h = self._calculate_entropy(self.current_belief)
        state = np.concatenate([self.current_belief, [h]])
        
        return state, reward, terminated, truncated, info

# ==============================================================================
# 2. COMPONENTES DQN (RED CUSTOM + BUFFER PER + AGENTE)
# ==============================================================================

class QNet(nn.Module):
    def __init__(self, state_space: int, action_space: int, net_arch: list):
        super(QNet, self).__init__()
        self.belief_dim = state_space - 1 
        self.entropy_dim = 1
        self.action_space = action_space
        
        # Rama del Belief (Arquitectura dinámica)
        layers = []
        input_dim = self.belief_dim
        for units in net_arch:
            layers.append(nn.Linear(input_dim, units))
            layers.append(nn.ReLU())
            input_dim = units
        self.belief_extractor = nn.Sequential(*layers)
        
        # Rama de Decisión (Inyección de Entropía)
        self.decision_head = nn.Sequential(
            nn.Linear(input_dim + self.entropy_dim, net_arch[-1]), # Última capa densa
            nn.ReLU(),
            nn.Linear(net_arch[-1], action_space)
        )

    def forward(self, x):
        belief = x[:, :-1] 
        entropy = x[:, -1:] 
        features = self.belief_extractor(belief)
        combined = torch.cat([features, entropy], dim=1)
        return self.decision_head(combined)

    def sample_action(self, obs, epsilon):
        if random.random() < epsilon:
            return random.randint(0, self.action_space - 1)
        else:
            with torch.no_grad():
                return self.forward(obs).argmax().item()

class PrioritizedReplayBuffer:
    def __init__(self, obs_dim, size, batch_size=32, alpha=0.6):
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size], dtype=np.float32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.priorities = np.zeros(size, dtype=np.float32)
        self.max_size, self.batch_size, self.ptr, self.size, self.alpha = size, batch_size, 0, 0, alpha

    def put(self, obs, act, rew, next_obs, done):
        max_prio = self.priorities.max() if self.size > 0 else 1.0
        self.priorities[self.ptr] = max_prio
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, beta=0.4):
        if self.size == 0: return None
        prios = self.priorities[:self.size]
        probs = prios ** self.alpha
        probs /= probs.sum()
        idxs = np.random.choice(self.size, self.batch_size, p=probs, replace=True)
        
        samples = dict(obs=self.obs_buf[idxs], next_obs=self.next_obs_buf[idxs],
                       acts=self.acts_buf[idxs], rews=self.rews_buf[idxs], done=self.done_buf[idxs])
        
        weights = (self.size * probs[idxs]) ** (-beta)
        weights /= weights.max()
        return samples, idxs, np.array(weights, dtype=np.float32)

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio + 1e-5

class DQNAgent:
    def __init__(self, state_dim, action_dim, h_params, device):
        self.device, self.gamma, self.batch_size, self.tau = device, h_params['gamma'], h_params['batch_size'], h_params['tau']
        
        net_arch = h_params.get('policy_kwargs', {}).get('net_arch', [128, 128])
        self.q_net = QNet(state_dim, action_dim, net_arch).to(device)
        self.target_q_net = QNet(state_dim, action_dim, net_arch).to(device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=h_params['learning_rate'])
        
        self.buffer = PrioritizedReplayBuffer(state_dim, h_params['buffer_size'], self.batch_size, h_params['alpha'])
        self.beta, self.beta_inc = h_params['beta_start'], (1.0 - h_params['beta_start']) / h_params['beta_frames']

    def store_transition(self, obs, act, rew, next_obs, done):
        self.buffer.put(obs, act, rew, next_obs, done)

    def update(self):
        if self.buffer.size < self.batch_size: return None
        self.beta = min(1.0, self.beta + self.beta_inc)
        samples, idxs, weights = self.buffer.sample(self.beta)
        
        states = torch.FloatTensor(samples['obs']).to(self.device)
        actions = torch.LongTensor(samples['acts'].reshape(-1,1)).to(self.device)
        rewards = torch.FloatTensor(samples['rews'].reshape(-1,1)).to(self.device)
        next_states = torch.FloatTensor(samples['next_obs']).to(self.device)
        dones = torch.FloatTensor(samples['done'].reshape(-1,1)).to(self.device)
        weights_t = torch.FloatTensor(weights).reshape(-1,1).to(self.device)

        with torch.no_grad():
            q_next = self.target_q_net(next_states).max(1)[0].unsqueeze(1)
            targets = rewards + self.gamma * q_next * (1 - dones)
        
        q_curr = self.q_net(states).gather(1, actions)
        loss = (F.smooth_l1_loss(q_curr, targets, reduction='none') * weights_t).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        td_errors = (targets - q_curr).abs().detach().cpu().numpy().squeeze()
        self.buffer.update_priorities(idxs, td_errors)
        return loss.item()

    def soft_update(self):
        for tp, lp in zip(self.target_q_net.parameters(), self.q_net.parameters()):
            tp.data.copy_(self.tau * lp.data + (1.0 - self.tau) * tp.data)
            
    def save(self, path): torch.save(self.q_net.state_dict(), path)
    def load(self, path): 
        self.q_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_q_net.load_state_dict(self.q_net.state_dict())

# ==============================================================================
# 3. UTILIDADES DE LOGGING Y PLOTTING
# ==============================================================================
class ExperimentLogger:
    def __init__(self, log_dir):
        self.path = os.path.join(log_dir, "monitor.csv")
        self.start = time.time()
        with open(self.path, 'w') as f: f.write(f"#{int(self.start)}\nsteps,r,l,t\n")
    def log(self, steps, r, l):
        with open(self.path, 'a') as f: f.write(f"{steps},{r},{l},{time.time() - self.start}\n")

# ... (El resto de funciones de plotting y utilidades se omiten por espacio, 
#      pero son las mismas que usaste anteriormente) ...

# ==============================================================================
# 4. EJECUCIÓN PRINCIPAL
# ==============================================================================
if __name__ == "__main__":
    
    # --- HIPERPARÁMETROS DEL DELAYED OBS ENV ---
    final_params = {
        'num_runs': 10,               
        'total_timesteps': 500_000, 
        
        # Parámetros del Entorno
        'env_size': 10,
        'slip_prob': 0.1,
        'delay_steps': 3, # Retardo de 3 pasos
        'max_episode_steps': 200,
        
        # Hiperparámetros de la DQN (v15 - estable)
        'learning_rate': 2.5e-5,     
        'buffer_size': 60_000,       
        'learning_starts': 20_000,   
        'batch_size': 2048,          
        'gamma': 0.95,               # Un poco más paciente para el delay
        'train_freq': (4, "step"),
        'tau': 0.005,
        'target_update_interval': 1, # Soft Update
        
        'exploration_initial_eps': 1.0,
        'exploration_final_eps': 0.001,
        'exploration_fraction': 0.3,
        
        # Arquitectura de la Red (Ajustada para 201 dimensiones de entrada)
        'policy_kwargs': dict(net_arch=[128, 128]), 
        
        'alpha': 0.6, 'beta_start': 0.4, 'beta_frames': 500_000
    }

    env_id = f"DelayedObsEnv-k{final_params['delay_steps', 'max_episode_steps']}"
    log_dir_base = f"IS-dqn_logs/{env_id}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # --- BUCLE DE EXPERIMENTOS ---
    for run_i in range(1, final_params['num_runs'] + 1):
        run_dir = f"{log_dir_base}/run_{run_i}"
        os.makedirs(run_dir, exist_ok=True)
        
        # Lógica para saltar runs existentes (omitida por brevedad)

        print(f"\n=== Iniciando {env_id} Run {run_i} ===")
        
        # 1. Instanciar Entorno BASE y WRAPPER
        base_env = DelayedStochasticObsEnv(
            size=final_params['env_size'],
            slip_prob=final_params['slip_prob'],
            delay_steps=final_params['delay_steps'],
        )
        env = DelayedBeliefWrapper(base_env) # <-- Aplicamos el Belief Wrapper

        # 2. Inicializar Agente
        state_dim = env.observation_space.shape[0] # 201
        action_dim = env.action_space.n # 4
        
        agent = DQNAgent(state_dim, action_dim, final_params, device)
        logger = ExperimentLogger(run_dir)
        
        # 3. Loop de Entrenamiento
        obs, _ = env.reset()
        ep_rew, ep_len = 0, 0
        eps_steps = final_params['total_timesteps'] * final_params['exploration_fraction']
        
        for t in range(1, final_params['total_timesteps'] + 1):
            
            # Epsilon decay
            eps = final_params['exploration_final_eps']
            if t < eps_steps:
                prog = t / eps_steps
                eps = 1.0 - prog * (1.0 - final_params['exploration_final_eps'])
            
            # Action
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
            action = agent.q_net.sample_action(obs_t, eps)
            
            # Step
            next_obs, rew, term, trunc, _ = env.step(action)
            done = term or trunc
            
            agent.store_transition(obs, action, rew, next_obs, done)
            
            obs = next_obs
            ep_rew += rew
            ep_len += 1
            
            # Train
            if t > final_params['learning_starts'] and t % final_params['train_freq'][0] == 0:
                agent.update()
                agent.soft_update() # Soft Update
            
            # Logging y Reset
            if done:
                logger.log(t, ep_rew, ep_len)
                obs, _ = env.reset()
                ep_rew, ep_len = 0, 0
                
            if t % 10000 == 0:
                print(f"Run {run_i} | Paso {t}")

        agent.save(f"{run_dir}/model.pth")
        env.close()
        print(f"Run {run_i} completado.")

    # --- Generación de Reportes Finales (omitiendo código) ---
    # generate_plots_and_tables(env_id, log_dir_base, final_params['num_runs'])