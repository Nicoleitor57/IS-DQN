# import gymnasium as gym
# import numpy as np
# import pandas as pd
# import time
# import os
# import glob
# import matplotlib.pyplot as plt
# import random
# from collections import deque
# import io

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim

# # --- 1. IMPORTAR ENTORNOS PERSONALIZADOS ---
# # (Asegúrate de que este archivo esté en la carpeta correcta para importar Entornos)
# try:
#     # from Entornos.TwoTigersEnv import TwoTigersEnv
#     # (Descomenta los otros si los vas a usar)
#     # from Entornos.PODoorEnv import POKeyDoorEnv
#     from Entornos.KeyDoorMazeEnv import KeyDoorMazeEnv
#     # from Entornos.DelayedObsEnv import DelayedObsEnv
# except ImportError:
#     print("Advertencia: No se pudieron importar los entornos desde la carpeta 'Entornos'.")
#     print("Asegúrate de que TwoTigersEnv.py esté en Entornos/")
#     exit()

# class KeyDoorBeliefWrapper(gym.Wrapper):
#     """
#     Wrapper PSR + Entropía para KeyDoorMazeEnv.
    
#     Estado Oculto (s): (x, y, estado_llave)
#         - x: 0..18
#         - y: 0..14
#         - estado_llave: 0 (Nada), 1 (Roja), 2 (Azul)
        
#     Total Estados: 19 * 15 * 3 = 855
    
#     Observación al Agente DQN:
#         Vector de 856 floats (855 probabilidades + 1 entropía).
#     """

#     def __init__(self, env):
#         super().__init__(env)
        
#         # Accedemos al entorno base para leer el mapa estático
#         self.base_env = env.unwrapped
        
#         self.H = self.base_env.height # 15
#         self.W = self.base_env.width  # 19
        
#         # Mapeo de llaves internas (env) a índices (belief)
#         # Env usa: 0=Nada, 2=Roja, 3=Azul
#         # Nosotros usamos índices: 0=Nada, 1=Roja, 2=Azul
#         self.key_map_env_to_idx = {0: 0, 2: 1, 3: 2}
#         self.key_map_idx_to_env = {0: 0, 1: 2, 2: 3}
        
#         self.num_key_states = 3
#         self.num_states = self.H * self.W * self.num_key_states # 855
        
#         # Copiamos datos estáticos del mapa para simular Z y T
#         self._static_grid = self.base_env._grid.copy() # 1=Muro, 0=Piso
#         self._key_red_pos = self.base_env._key_red_pos
#         self._key_blue_pos = self.base_env._key_blue_pos
#         self._door_red_pos = self.base_env._door_red_pos
#         self._door_blue_pos = self.base_env._door_blue_pos
#         self._door_red_trap = self.base_env._door_red_pos_trap
#         self._door_blue_trap = self.base_env._door_blue_pos_trap
        
#         # Definir espacio de observación del Wrapper
       
#         self.observation_space = gym.spaces.Box(
#             low=0.0,
#             high=1.0, # Ahora normalizado
#             shape=(self.num_states + 1,), 
#             dtype=np.float32
#         )
        
#         # Inicializar belief
#         self.current_belief = np.zeros(self.num_states, dtype=np.float32)

#     # --- Utilitarios de Índices ---
    
#     def _state_to_idx(self, x, y, k_idx):
#         # k_idx: 0, 1, 2
#         return k_idx * (self.H * self.W) + x * self.W + y

#     def _idx_to_state(self, idx):
#         k_idx = idx // (self.H * self.W)
#         rem = idx % (self.H * self.W)
#         x = rem // self.W
#         y = rem % self.W
#         return x, y, k_idx

#     # --- Lógica del Filtro Bayesiano ---

#     def _predict(self, belief, action):
#         """
#         Aplica la función de transición determinista f(s, a)
#         """
#         new_belief = np.zeros_like(belief)
        
#         # Iteramos solo sobre los estados con probabilidad > 0 (Optimización masiva)
#         active_indices = np.where(belief > 0.0)[0]
        
#         for idx in active_indices:
#             prob = belief[idx]
#             x, y, k_idx = self._idx_to_state(idx)
            
#             # 1. Calcular Movimiento (Copia de lógica del env)
#             dx, dy = [(-1, 0), (1, 0), (0, -1), (0, 1)][action]
#             nx, ny = x + dx, y + dy
            
#             # Colisión con muro (Grid estático tiene 1 en muros)
#             if self._static_grid[nx, ny] == 1:
#                 nx, ny = x, y # Se queda quieto
            
#             # 2. Calcular Cambio de Llave
#             nk_idx = k_idx
            
#             # Si no tiene llave (k_idx=0), ver si pisa una
#             if k_idx == 0:
#                 if (nx, ny) == self._key_red_pos:
#                     nk_idx = 1 # Coge Roja
#                 elif (nx, ny) == self._key_blue_pos:
#                     nk_idx = 2 # Coge Azul
            
#             # 3. Mover probabilidad al nuevo estado
#             # Nota: No modelamos "Terminar" aquí porque el step real nos dirá si terminó.
#             # Si el episodio sigue, el estado evoluciona así.
#             new_idx = self._state_to_idx(nx, ny, nk_idx)
#             new_belief[new_idx] += prob
            
#         return new_belief

#     def _generate_expected_obs(self, x, y, k_idx):
#         """
#         Genera la observación de 3x3 exacta que vería el agente
#         estando en (x,y) con el estado de llave k_idx.
#         Simula la función g(s).
#         """
#         # Reconstruimos el grid lógico para este estado hipotético
#         # Empezamos con el estático (Muros=1, Pisos=0)
#         # Ojo: El env dinámico pone las llaves y puertas sobre esto.
        
#         # Creamos una vista local temporal
#         # Para optimizar, no reconstruimos todo el grid 15x19, solo el patch 3x3
#         obs = np.ones((3, 3), dtype=np.int8) # Default a algo (no importa, se sobreescribe)
        
#         # Lógica de renderizado del env:
#         # Values: 0:Piso, 1:Muro, 2:KR, 3:KB, 4:DR, 5:DB
        
#         for i in range(3):
#             for j in range(3):
#                 grid_x = x - 1 + i
#                 grid_y = y - 1 + j
                
#                 val = 1 # Default Muro (si fuera de límites)
                
#                 if 0 <= grid_x < self.H and 0 <= grid_y < self.W:
#                     # 1. Base Muro/Piso
#                     val = self._static_grid[grid_x, grid_y]
                    
#                     pos = (grid_x, grid_y)
                    
#                     # 2. Objetos dinámicos (Llaves)
#                     # Si k_idx=0, las llaves están en el mapa.
#                     # Si k_idx=1, roja no está. Si k_idx=2, azul no está.
#                     if pos == self._key_red_pos and k_idx != 1:
#                         val = 2
#                     elif pos == self._key_blue_pos and k_idx != 2:
#                         val = 3
                    
#                     # 3. Puertas (Siempre están, a menos que se abran, pero al abrir acaba el ep)
#                     # Así que siempre las dibujamos
#                     if pos == self._door_red_pos: val = 4
#                     elif pos == self._door_blue_pos: val = 5
#                     elif pos == self._door_red_trap: val = 4
#                     elif pos == self._door_blue_trap: val = 5
                
#                 obs[i, j] = val
        
#         # 4. El agente siempre se ve a sí mismo en el centro
#         obs[1, 1] = 6
#         return obs

#     def _update(self, predicted_belief, real_obs):
#         """
#         Actualiza comparando la observación real con la esperada g(s).
#         """
#         new_belief = np.zeros_like(predicted_belief)
#         active_indices = np.where(predicted_belief > 0.0)[0]
        
#         consistent_prob_sum = 0.0
        
#         for idx in active_indices:
#             x, y, k_idx = self._idx_to_state(idx)
            
#             expected_obs = self._generate_expected_obs(x, y, k_idx)
            
#             if np.array_equal(real_obs, expected_obs):
#                 new_belief[idx] = predicted_belief[idx]
#                 consistent_prob_sum += predicted_belief[idx]
#             else:
#                 new_belief[idx] = 0.0
        
#         # Normalizar
#         if consistent_prob_sum < 1e-9:
#             # El modelo colapsó (observación imposible según creencia).
#             # Fallback: Inyección de incertidumbre o mantener predicción (arriesgado)
#             # O volver al prior uniforme local.
#             # En un entorno determinista perfecto, esto no debería pasar si b0 es correcto.
#             # Si pasa, retornamos uniforme sobre todo el mapa para recuperar.
#             return np.ones(self.num_states, dtype=np.float32) / self.num_states
        
#         return new_belief / consistent_prob_sum

#     def _calculate_entropy(self, belief):
#         """Calcula la entropía de Shannon NORMALIZADA (0 a 1)."""
#         belief = np.clip(belief, 1e-9, 1.0)
#         entropy = -np.sum(belief * np.log2(belief))
        
#         # Normalización: Dividir por la entropía máxima posible (log2 del número de estados)
#         # log2(855) es aprox 9.74. 
#         # Si H=0 (certeza total) -> 0.0
#         # Si H=Max (incertidumbre total) -> 1.0
#         max_entropy = np.log2(self.num_states)
#         normalized_entropy = entropy / max_entropy
        
#         return np.float32(normalized_entropy)

#     # --- Métodos Gym ---

#     def reset(self, **kwargs):
#         obs, info = self.env.reset(**kwargs)
        
#         # Reiniciar Belief
#         # Conocemos la posición inicial y que no tenemos llave
#         start_x, start_y = self.base_env._start_pos
#         start_idx = self._state_to_idx(start_x, start_y, 0) # k=0
        
#         self.current_belief = np.zeros(self.num_states, dtype=np.float32)
#         self.current_belief[start_idx] = 1.0
        
#         # Calcular entropía (será 0.0 al inicio)
#         h = self._calculate_entropy(self.current_belief)
        
#         return np.concatenate([self.current_belief, [h]]), info

#     def step(self, action):
#         # 1. Paso real
#         real_obs, reward, terminated, truncated, info = self.env.step(action)
        
#         # 2. Filtro Bayesiano
#         # a) Predicción f(b, a)
#         pred_belief = self._predict(self.current_belief, action)
        
#         # b) Actualización con Z(o|s)
#         self.current_belief = self._update(pred_belief, real_obs)
        
#         # 3. Salida
#         h = self._calculate_entropy(self.current_belief)
#         state = np.concatenate([self.current_belief, [h]])
        
#         return state, reward, terminated, truncated, info
    

# class QNet(nn.Module):
#     """
#     Red Especializada: Procesa el Belief y la Entropía por separado
#     y los une para la decisión final.
#     """
#     def __init__(self, state_space: int, action_space: int, net_arch: list = [128, 128]):
#         super(QNet, self).__init__()
        
#         # Asumimos que el último valor del state_space es la Entropía
#         self.belief_dim = state_space - 1 
#         self.entropy_dim = 1
#         self.action_space = action_space
        
#         # --- Rama 1: Procesamiento del Belief (Extracción de características) ---
#         # Esta capa comprime el mapa de 855 estados a una representación densa
#         self.belief_extractor = nn.Sequential(
#             nn.Linear(self.belief_dim, 128),
#             nn.ReLU(),
#             nn.Linear(128, 128),
#             nn.ReLU()
#         )
        
#         # --- Rama 2: Decisión Final (Combinación) ---
#         # La entrada aquí es: 128 (features del belief) + 1 (entropía cruda)
#         combined_dim = 128 + self.entropy_dim
        
#         self.decision_head = nn.Sequential(
#             nn.Linear(combined_dim, 64),
#             nn.ReLU(),
#             nn.Linear(64, action_space)
#         )

#     def forward(self, x):
#         # x tiene shape (batch_size, 856)
        
#         # 1. Separar el Belief y la Entropía
#         # Belief son todas las columnas menos la última
#         belief = x[:, :-1] 
#         # Entropía es la última columna
#         entropy = x[:, -1:] 
        
#         # 2. Procesar el Belief
#         belief_features = self.belief_extractor(belief)
        
#         # 3. Concatenar: Inyectamos la entropía directamente en la capa de decisión
#         # Esto evita que la entropía se "diluya" en las primeras capas
#         combined = torch.cat([belief_features, entropy], dim=1)
        
#         # 4. Calcular Q-values
#         q_values = self.decision_head(combined)
        
#         return q_values

#     # (El resto de métodos como sample_action se mantienen igual)
#     def sample_action(self, obs, epsilon: float):
#         if random.random() < epsilon:
#             return random.randint(0, self.action_space - 1)
#         else:
#             with torch.no_grad():
#                 return self.forward(obs).argmax().item()

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

# --- IMPORTACIÓN DE ENTORNOS ---
try:
    from Entornos.KeyDoorMazeEnv import KeyDoorMazeEnv
except ImportError:
    print("ERROR CRÍTICO: No se encuentra 'Entornos/KeyDoorMazeEnv.py'.")
    exit()

# ==============================================================================
# 1. WRAPPER: BELIEF STATE (PSR) + ENTROPY
# ==============================================================================
class KeyDoorBeliefWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.base_env = env.unwrapped
        self.H = self.base_env.height
        self.W = self.base_env.width
        
        # 3 estados de llave: 0(Nada), 1(Roja), 2(Azul)
        self.num_key_states = 3
        self.num_states = self.H * self.W * self.num_key_states # 855
        
        # Copias estáticas para el modelo interno
        self._static_grid = self.base_env._grid.copy()
        self._key_red_pos = self.base_env._key_red_pos
        self._key_blue_pos = self.base_env._key_blue_pos
        self._door_red_pos = self.base_env._door_red_pos
        self._door_blue_pos = self.base_env._door_blue_pos
        self._door_red_trap = self.base_env._door_red_pos_trap
        self._door_blue_trap = self.base_env._door_blue_pos_trap
        
        # Espacio: [Belief (855) + Entropía (1)]
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(self.num_states + 1,), dtype=np.float32
        )
        self.current_belief = np.zeros(self.num_states, dtype=np.float32)

    def _state_to_idx(self, x, y, k_idx):
        return k_idx * (self.H * self.W) + x * self.W + y

    def _idx_to_state(self, idx):
        area = self.H * self.W
        k_idx = idx // area
        rem = idx % area
        x = rem // self.W
        y = rem % self.W
        return x, y, k_idx

    def _calculate_entropy(self, belief):
        """Entropía Normalizada (0 a 1)"""
        b = np.clip(belief, 1e-9, 1.0)
        entropy = -np.sum(b * np.log2(b))
        max_entropy = np.log2(self.num_states)
        return np.float32(entropy / max_entropy)

    def _generate_expected_obs(self, x, y, k_idx):
        """Simula la función de observación g(s)"""
        obs = np.ones((3, 3), dtype=np.int8)
        for i in range(3):
            for j in range(3):
                grid_x, grid_y = x - 1 + i, y - 1 + j
                val = 1 # Muro default
                if 0 <= grid_x < self.H and 0 <= grid_y < self.W:
                    val = self._static_grid[grid_x, grid_y]
                    pos = (grid_x, grid_y)
                    if pos == self._key_red_pos and k_idx != 1: val = 2
                    elif pos == self._key_blue_pos and k_idx != 2: val = 3
                    if pos == self._door_red_pos: val = 4
                    elif pos == self._door_blue_pos: val = 5
                    elif pos == self._door_red_trap: val = 4
                    elif pos == self._door_blue_trap: val = 5
                obs[i, j] = val
        obs[1, 1] = 6
        return obs

    def _predict(self, belief, action):
        """Transición f(b, a)"""
        new_belief = np.zeros_like(belief)
        active = np.where(belief > 0.0)[0]
        for idx in active:
            prob = belief[idx]
            x, y, k = self._idx_to_state(idx)
            
            # Dinámica de movimiento
            dx, dy = [(-1, 0), (1, 0), (0, -1), (0, 1)][action]
            nx, ny = x + dx, y + dy
            if self._static_grid[nx, ny] == 1: nx, ny = x, y
            
            # Dinámica de llave
            nk = k
            if k == 0:
                if (nx, ny) == self._key_red_pos: nk = 1
                elif (nx, ny) == self._key_blue_pos: nk = 2
            
            new_idx = self._state_to_idx(nx, ny, nk)
            new_belief[new_idx] += prob
        return new_belief

    def _update(self, belief, real_obs):
        """Actualización bayesiana"""
        new_belief = np.zeros_like(belief)
        active = np.where(belief > 0.0)[0]
        sum_prob = 0.0
        for idx in active:
            x, y, k = self._idx_to_state(idx)
            expected = self._generate_expected_obs(x, y, k)
            if np.array_equal(real_obs, expected):
                new_belief[idx] = belief[idx]
                sum_prob += belief[idx]
        
        if sum_prob < 1e-9: # Colapso del filtro
             return np.ones(self.num_states, dtype=np.float32) / self.num_states
        return new_belief / sum_prob

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        sx, sy = self.base_env._start_pos
        start_idx = self._state_to_idx(sx, sy, 0)
        self.current_belief = np.zeros(self.num_states, dtype=np.float32)
        self.current_belief[start_idx] = 1.0
        h = self._calculate_entropy(self.current_belief)
        return np.concatenate([self.current_belief, [h]]), info

    def step(self, action):
        real_obs, reward, term, trunc, info = self.env.step(action)
        pred = self._predict(self.current_belief, action)
        self.current_belief = self._update(pred, real_obs)
        h = self._calculate_entropy(self.current_belief)
        return np.concatenate([self.current_belief, [h]]), reward, term, trunc, info

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
        # Toma la salida de la rama belief + 1 (entropía)
        self.decision_head = nn.Sequential(
            nn.Linear(input_dim + self.entropy_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_space)
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
        idxs = np.random.choice(self.size, self.batch_size, p=probs)
        
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
        self.device = device
        self.gamma = h_params['gamma']
        self.batch_size = h_params['batch_size']
        self.tau = h_params['tau']
        
        # Red Neuronal
        net_arch = h_params.get('policy_kwargs', {}).get('net_arch', [256, 256])
        self.q_net = QNet(state_dim, action_dim, net_arch).to(device)
        self.target_q_net = QNet(state_dim, action_dim, net_arch).to(device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=h_params['learning_rate'])
        
        # Buffer de Repetición
        self.buffer = PrioritizedReplayBuffer(state_dim, h_params['buffer_size'], self.batch_size, h_params['alpha'])
        self.beta, self.beta_inc = h_params['beta_start'], (1.0 - h_params['beta_start']) / h_params['beta_frames']

    # --- ¡MÉTODO AGREGADO! ---
    def store_transition(self, obs, act, rew, next_obs, done):
        """Almacena la transición en el buffer interno."""
        self.buffer.put(obs, act, rew, next_obs, done)
    # -------------------------

    def update(self):
        if self.buffer.size < self.batch_size: return None
        
        # Annealing de Beta
        self.beta = min(1.0, self.beta + self.beta_inc)
        
        # Muestreo del Buffer
        samples, idxs, weights = self.buffer.sample(self.beta)
        
        states = torch.FloatTensor(samples['obs']).to(self.device)
        actions = torch.LongTensor(samples['acts'].reshape(-1,1)).to(self.device)
        rewards = torch.FloatTensor(samples['rews'].reshape(-1,1)).to(self.device)
        next_states = torch.FloatTensor(samples['next_obs']).to(self.device)
        dones = torch.FloatTensor(samples['done'].reshape(-1,1)).to(self.device)
        weights_t = torch.FloatTensor(weights).reshape(-1,1).to(self.device)

        # Cálculo del Target (Double DQN logic implícita al usar target network)
        with torch.no_grad():
            q_next = self.target_q_net(next_states).max(1)[0].unsqueeze(1)
            targets = rewards + self.gamma * q_next * (1 - dones)
        
        # Cálculo del Q actual
        q_curr = self.q_net(states).gather(1, actions)
        
        # Loss ponderada por pesos de PER
        loss = (F.smooth_l1_loss(q_curr, targets, reduction='none') * weights_t).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Actualizar prioridades en el buffer
        td_errors = (targets - q_curr).abs().detach().cpu().numpy().squeeze()
        self.buffer.update_priorities(idxs, td_errors)
        
        return loss.item()

    def soft_update(self):
        for tp, lp in zip(self.target_q_net.parameters(), self.q_net.parameters()):
            tp.data.copy_(self.tau * lp.data + (1.0 - self.tau) * tp.data)
            
    def save(self, path): 
        torch.save(self.q_net.state_dict(), path)
    
    def sample_action(self, obs, epsilon):
        # Delegamos a la red interna
        return self.q_net.sample_action(obs, epsilon)
    

# ==============================================================================
# 3. UTILIDADES DE LOGGING Y PLOTTING
# ==============================================================================
class ExperimentLogger:
    def __init__(self, log_dir):
        self.path = os.path.join(log_dir, "monitor.csv")
        self.start = time.time()
        with open(self.path, 'w') as f: f.write(f"#{int(self.start)}\nr,l,t\n")
    def log(self, r, l):
        with open(self.path, 'a') as f: f.write(f"{r},{l},{time.time() - self.start}\n")

def generate_plots_and_tables(env_id, log_base_dir, num_runs):
    print(f"\n--- Generando reportes para {env_id} ---")
    all_rewards, all_lengths = [], []
    min_len = float('inf')

    # 1. Cargar datos
    for i in range(1, num_runs + 1):
        file = f"{log_base_dir}/run_{i}/monitor.csv"
        if os.path.exists(file):
            try:
                df = pd.read_csv(file, skiprows=1)
                if not df.empty:
                    all_rewards.append(df['r'].values)
                    all_lengths.append(df['l'].values)
                    min_len = min(min_len, len(df))
            except: pass
    
    if not all_rewards:
        print("No hay datos válidos para graficar.")
        return

    # 2. Truncar y Promediar
    rewards = np.array([r[:min_len] for r in all_rewards])
    lengths = np.array([l[:min_len] for l in all_lengths])
    
    avg_rewards = rewards.mean(axis=0)
    std_rewards = rewards.std(axis=0)
    avg_lengths = lengths.mean(axis=0)
    
    # Suavizado para gráficos
    window = max(1, min_len // 20)
    smooth_rew = pd.Series(avg_rewards).rolling(window, min_periods=1).mean()
    smooth_len = pd.Series(avg_lengths).rolling(window, min_periods=1).mean()

    # 3. Graficar Recompensa
    plt.figure(figsize=(10, 6))
    plt.plot(avg_rewards, alpha=0.3, color='gray', label='Promedio crudo')
    plt.plot(smooth_rew, color='green', linewidth=2, label=f'Media móvil (v={window})')
    plt.fill_between(range(min_len), smooth_rew - std_rewards, smooth_rew + std_rewards, color='green', alpha=0.1)
    plt.title(f"Recompensa Promedio - {env_id}")
    plt.xlabel("Episodios"); plt.ylabel("Recompensa")
    plt.legend(); plt.grid(True)
    plt.savefig(f"{env_id}_rewards.png")
    print(f"Gráfico guardado: {env_id}_rewards.png")

    # 4. Graficar Longitud
    plt.figure(figsize=(10, 6))
    plt.plot(smooth_len, color='blue', linewidth=2)
    plt.title(f"Largo Promedio - {env_id}")
    plt.xlabel("Episodios"); plt.ylabel("Pasos")
    plt.grid(True)
    plt.savefig(f"{env_id}_lengths.png")
    print(f"Gráfico guardado: {env_id}_lengths.png")

    # 5. Tabla Resumen
    print("\nTABLA DE RESULTADOS PROMEDIO:")
    print(f"{'Rango Episodios':<20} | {'Recompensa Media':<20} | {'Largo Medio':<20}")
    print("-" * 66)
    step = max(1, min_len // 10)
    for i in range(0, min_len, step):
        end = min(i + step, min_len)
        r_mean = np.mean(avg_rewards[i:end])
        l_mean = np.mean(avg_lengths[i:end])
        print(f"{f'{i}-{end}':<20} | {r_mean:<20.2f} | {l_mean:<20.2f}")

# ==============================================================================
# 4. EJECUCIÓN PRINCIPAL
# ==============================================================================
if __name__ == "__main__":
    
    # --- HIPERPARÁMETROS KEYDOOR MAZE ---
    # Ajustados para el nuevo entorno de 855 estados
    final_params = {
        'num_runs': 10,               # 5 Corridas para probar
        'total_timesteps': 300_000,  # Más pasos porque el maze es más difícil
        'learning_rate': 2.5e-5,     
        'buffer_size': 100_000,      
        'learning_starts': 20_000,   # Empezar antes para ver si avanza
        'batch_size': 2048,           # Estabilidad
        'gamma': 0.95,               # Un poco más paciente para el laberinto largo
        'train_freq': (4, "step"),
        'tau': 0.005,
        'target_update_interval': 1,
        
        'exploration_initial_eps': 1.0,
        'exploration_final_eps': 0.01,
        'exploration_fraction': 0.3, # Explorar el 30% del tiempo (laberinto grande)
        
        # RED MÁS GRANDE PARA INPUT DE 856 DIMENSIONES
        'policy_kwargs': dict(net_arch=[512, 512]), 
        
        'alpha': 0.6, 'beta_start': 0.4, 'beta_frames': 500_000
    }

    env_id = "KeyDoorMazeEnv"
    log_dir_base = f"IS-dqn_logs/{env_id}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # --- BUCLE DE EXPERIMENTOS ---
    for run_i in range(1, final_params['num_runs'] + 1):
        run_dir = f"{log_dir_base}/run_{run_i}"
        os.makedirs(run_dir, exist_ok=True)
        
        if os.path.exists(f"{run_dir}/monitor.csv"):
            print(f"Run {run_i} ya existe. Saltando.")
            continue

        print(f"\n=== Iniciando {env_id} Run {run_i} ===")
        
        # 1. Init Entorno y Agente
        env = KeyDoorMazeEnv(max_episode_steps=1000) # Horizonte más largo para el laberinto
        env = KeyDoorBeliefWrapper(env)
        
        agent = DQNAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
            h_params=final_params,
            device=device
        )
        logger = ExperimentLogger(run_dir)
        
        # 2. Loop de Entrenamiento
        obs, _ = env.reset()
        ep_rew, ep_len = 0, 0
        
        # Epsilon schedule
        eps_steps = final_params['total_timesteps'] * final_params['exploration_fraction']
        
        start_t = time.time()
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
                agent.soft_update()
            
            # Logging
            if done:
                logger.log(ep_rew, ep_len)
                obs, _ = env.reset()
                ep_rew, ep_len = 0, 0
            
            if t % 10000 == 0:
                print(f"Run {run_i} | Paso {t} | Eps: {eps:.3f}")

        agent.save(f"{run_dir}/model.pth")
        env.close()
        print(f"Run {run_i} completado en {(time.time()-start_t)/60:.1f} min")

    # --- GENERAR REPORTES AL FINAL ---
    generate_plots_and_tables(env_id, log_dir_base, final_params['num_runs'])
    plt.show()