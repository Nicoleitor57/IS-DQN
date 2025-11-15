# import torch
# import torch.nn as nn
# from torch.nn import functional as F

# # Parámetros para TwoTigersEnv
# NUM_OBS_CATEGORIES_1 = 3  # [0: Silencio, 1: Sonido Izq, 2: Sonido Der]
# NUM_OBS_CATEGORIES_2 = 3  # [0: Silencio, 1: Sonido Izq, 2: Sonido Der]
# NUM_ACTIONS = 9           # spaces.Discrete(9)

# EMBED_DIM_OBS = 8         # Dimensión para embeber cada observación
# EMBED_DIM_ACT = 8         # Dimensión para embeber la acción
# RNN_HIDDEN_DIM = 64       # Tamaño del estado oculto (nuestro PSR 'ψ_h')

# class PSR_DRQN_Network(nn.Module):
#     def __init__(self):
#         super().__init__()
        
#         # 1. Capas de Embedding
#         # Embebemos las dos observaciones categóricas
#         self.embed_obs_1 = nn.Embedding(NUM_OBS_CATEGORIES_1, EMBED_DIM_OBS)
#         self.embed_obs_2 = nn.Embedding(NUM_OBS_CATEGORIES_2, EMBED_DIM_OBS)
        
#         # Embebemos la acción *previa* (importante para el historial)
#         self.embed_prev_action = nn.Embedding(NUM_ACTIONS, EMBED_DIM_ACT)
        
#         # Calculamos el tamaño de entrada para el RNN
#         rnn_input_dim = (EMBED_DIM_OBS * 2) + EMBED_DIM_ACT
        
#         # 2. Cuerpo Recurrente (El codificador de historial)
#         # Usamos GRU que es un poco más ligero que LSTM
#         self.rnn = nn.GRU(rnn_input_dim, RNN_HIDDEN_DIM, batch_first=True)
        
#         # 3. CABEZAS (Salidas)
        
#         # Cabeza 1: Q-Value (Tarea principal)
#         # Toma h_t -> predice Q(h_t, a) para las 9 acciones
#         self.q_head = nn.Linear(RNN_HIDDEN_DIM, NUM_ACTIONS)
        
#         # Cabeza 2: PSR - Predicción de Recompensa (Tarea Auxiliar 1)
#         # Toma h_t -> predice r_t (un valor escalar)
#         self.psr_reward_head = nn.Linear(RNN_HIDDEN_DIM, 1)
        
#         # Cabeza 3: PSR - Predicción de Próxima Observación (Tarea Auxiliar 2)
#         # Toma h_t -> predice los logits para o_{t+1}
#         self.psr_obs_1_head = nn.Linear(RNN_HIDDEN_DIM, NUM_OBS_CATEGORIES_1)
#         self.psr_obs_2_head = nn.Linear(RNN_HIDDEN_DIM, NUM_OBS_CATEGORIES_2)

#     def forward(self, obs, prev_actions, hidden_state):
#         # obs shape: (batch_size, sequence_length, 2)
#         # prev_actions shape: (batch_size, sequence_length)
#         # hidden_state shape: (1, batch_size, RNN_HIDDEN_DIM)
        
#         # Embeber entradas
#         # obs[:, :, 0] es la obs 1 de toda la secuencia
#         # obs[:, :, 1] es la obs 2 de toda la secuencia
#         emb_obs_1 = self.embed_obs_1(obs[:, :, 0])
#         emb_obs_2 = self.embed_obs_2(obs[:, :, 1])
#         emb_prev_act = self.embed_prev_action(prev_actions)
        
#         # Concatenar para formar la entrada del RNN
#         rnn_input = torch.cat([emb_obs_1, emb_obs_2, emb_prev_act], dim=2)
        
#         # Procesar con el RNN
#         # rnn_output es la secuencia de estados ocultos (nuestros PSRs 'ψ_h')
#         # hidden_state es el *último* estado oculto
#         rnn_output, new_hidden_state = self.rnn(rnn_input, hidden_state)
        
#         # Pasar el *historial* (rnn_output) por las cabezas
        
#         # 1. Q-Values
#         q_values = self.q_head(rnn_output)
        
#         # 2. Predicciones PSR (recompensa)
#         pred_rewards = self.psr_reward_head(rnn_output)
        
#         # 3. Predicciones PSR (próxima observación)
#         pred_obs_1_logits = self.psr_obs_1_head(rnn_output)
#         pred_obs_2_logits = self.psr_obs_2_head(rnn_output)
        
#         # Devolvemos todo
#         return {
#             "q_values": q_values,              # Para L_DQN
#             "pred_rewards": pred_rewards,        # Para L_PSR
#             "pred_obs_1_logits": pred_obs_1_logits, # Para L_PSR
#             "pred_obs_2_logits": pred_obs_2_logits, # Para L_PSR
#             "hidden_state": new_hidden_state     # Para el siguiente paso
#         }
        
        
# # --- Parámetros de Pérdida ---
# BETA_REWARD = 0.3  # Hiperparámetro (β) para la pérdida de recompensa
# BETA_OBS = 0.5     # Hiperparámetro (β) para la pérdida de observación
# GAMMA = 0.99       # Factor de descuento

# # --- Criterios de Pérdida ---
# loss_fn_dqn = nn.MSELoss()
# loss_fn_psr_reward = nn.MSELoss()
# loss_fn_psr_obs = nn.CrossEntropyLoss()

# def calculate_combined_loss(batch, main_network, target_network):
#     """
#     Calcula la pérdida combinada L_Total = L_DQN + (β * L_PSR)
    
#     Asumimos que 'batch' es un diccionario con secuencias:
#     batch = {
#         "obs": tensor shape (B, T, 2), # Secuencia de observaciones
#         "actions": tensor shape (B, T),   # Secuencia de acciones
#         "rewards": tensor shape (B, T),   # Secuencia de recompensas
#         "next_obs": tensor shape (B, T, 2),# Secuencia de sig. observaciones
#         "dones": tensor shape (B, T)      # Secuencia de 'dones'
#     }
#     Donde B=batch_size, T=sequence_length
#     """
    
#     # 0. Preparar datos
#     # Necesitamos la acción *previa* para el RNN
#     # (El primer paso usa un "dummy" action_pad, ej. un tensor de ceros)
#     prev_actions = ... # Lógica para obtener acciones previas del batch
    
#     # Obtenemos el estado oculto inicial (normalmente ceros)
#     initial_hidden_state = ... # shape (1, batch_size, RNN_HIDDEN_DIM)

#     # 1. Forward pass en la RED PRINCIPAL
#     # Obtenemos predicciones para TODA la secuencia
#     main_preds = main_network(batch["obs"], prev_actions, initial_hidden_state)
    
#     # Q-values predichos para las acciones tomadas
#     # main_preds["q_values"] tiene shape (B, T, 9)
#     # batch["actions"] tiene shape (B, T)
#     q_values_predicted = torch.gather(main_preds["q_values"], 2, batch["actions"].unsqueeze(-1)).squeeze(-1)

#     # 2. Calcular el Target de Bellman (L_DQN)
#     with torch.no_grad():
#         # Obtenemos Q-values de la red TARGET para el *siguiente* estado
#         # Nota: La propagación del estado oculto aquí es crucial
#         next_preds_target = target_network(batch["next_obs"], batch["actions"], initial_hidden_state)
        
#         # Double-DQN:
#         # 1. Encontrar la mejor acción usando la red *principal*
#         next_preds_main = main_network(batch["next_obs"], batch["actions"], initial_hidden_state)
#         best_next_actions = torch.argmax(next_preds_main["q_values"], dim=2).unsqueeze(-1)
        
#         # 2. Obtener el Q-value de esa acción desde la red *target*
#         q_next_target = torch.gather(next_preds_target["q_values"], 2, best_next_actions).squeeze(-1)

#         # Calcular el target Y_t
#         q_target = batch["rewards"] + GAMMA * (1 - batch["dones"]) * q_next_target

#     # ---- PÉRDIDA 1: L_DQN ----
#     loss_dqn = loss_fn_dqn(q_values_predicted, q_target)

#     # 3. Calcular Pérdidas Auxiliares (L_PSR)
    
#     # ---- PÉRDIDA 2: L_PSR (Recompensa) ----
#     # predecir r_t usando h_t
#     pred_rewards = main_preds["pred_rewards"].squeeze(-1) # Shape (B, T)
#     true_rewards = batch["rewards"] # Shape (B, T)
#     loss_psr_reward = loss_fn_psr_reward(pred_rewards, true_rewards)
    
#     # ---- PÉRDIDA 3: L_PSR (Observación) ----
#     # predecir o_{t+1} usando h_t
    
#     # Requerimos que las salidas (logits) estén en shape (B*T, Num_Clases)
#     # y los targets en shape (B*T)
#     B, T, _ = main_preds["pred_obs_1_logits"].shape
    
#     pred_obs_1_logits = main_preds["pred_obs_1_logits"].view(B * T, NUM_OBS_CATEGORIES_1)
#     pred_obs_2_logits = main_preds["pred_obs_2_logits"].view(B * T, NUM_OBS_CATEGORIES_2)
    
#     # Usamos batch["next_obs"] como el target
#     true_next_obs_1 = batch["next_obs"][:, :, 0].view(B * T)
#     true_next_obs_2 = batch["next_obs"][:, :, 1].view(B * T)

#     loss_psr_obs_1 = loss_fn_psr_obs(pred_obs_1_logits, true_next_obs_1)
#     loss_psr_obs_2 = loss_fn_psr_obs(pred_obs_2_logits, true_next_obs_2)
    
#     loss_psr_obs = loss_psr_obs_1 + loss_psr_obs_2
    
#     # 4. Calcular Pérdida Total
#     loss_total = loss_dqn + (BETA_REWARD * loss_psr_reward) + (BETA_OBS * loss_psr_obs)
    
#     # 5. Devolver pérdidas (para logging y backpropagation)
#     return loss_total, loss_dqn, loss_psr_reward, loss_psr_obs

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import time
import os
import io
import matplotlib.pyplot as plt
from collections import deque, namedtuple
from tqdm import trange


# ==============================================================================
# === 1. ENTORNO (Tu código de TwoTigersEnv) ====================================
# ==============================================================================

class TwoTigersEnv(gym.Env):
    """
    Entorno de los Dos Tigres Factoreados (Compatible con Gymnasium)
    """

    metadata = {"render_modes": ["ansi", "human"]}

    def __init__(self, accuracy=0.85, max_episode_steps=200, render_mode="ansi"):
        super().__init__()
        
        self.accuracy = accuracy
        self.max_episode_steps = max_episode_steps
        self._render_mode = render_mode
        
        self.R_TREASURE = 10
        self.R_TIGER = -100
        self.R_LISTEN = -1

        # Acción discreta (0-8) que codifica [act1, act2]
        self.action_space = spaces.Discrete(9) 
        # Observación [obs1, obs2]
        self.observation_space = spaces.MultiDiscrete([3, 3])

        # Estado interno
        self._tiger_pos_1 = 0
        self._tiger_pos_2 = 0
        self._current_steps = 0
        
        self._last_action = None
        self._last_observation = None
        self._last_reward = 0
        
    def _get_info(self):
        return {
            "tiger_1_pos": "Left" if self._tiger_pos_1 == 0 else "Right",
            "tiger_2_pos": "Left" if self._tiger_pos_2 == 0 else "Right",
            "steps": self._current_steps
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self._tiger_pos_1 = self.np_random.integers(0, 2)
        self._tiger_pos_2 = self.np_random.integers(0, 2)
        self._current_steps = 0
        
        # [0, 0] es "Silencio" para ambos
        observation = np.array([0, 0], dtype=np.int64)
        info = self._get_info()
        
        self._last_action = None
        self._last_observation = observation
        self._last_reward = 0
        
        if self._render_mode == "human":
            self.render()
        
        return observation, info

    def step(self, action):
        # Decodificar la acción Discreta(9) en MultiDiscrete([3, 3])
        act_1 = action // 3  # División entera (0-2)
        act_2 = action % 3   # Módulo (0-2)
        
        total_reward = 0
        obs_1 = 0
        obs_2 = 0

        # --- Lógica del Problema 1 ---
        if act_1 == 0: # Escuchar
            total_reward += self.R_LISTEN
            obs_1 = (1 if self._tiger_pos_1 == 0 else 2) if self.np_random.random() < self.accuracy else (2 if self._tiger_pos_1 == 0 else 1)
        elif act_1 == 1: # Abrir Izquierda
            total_reward += self.R_TIGER if self._tiger_pos_1 == 0 else self.R_TREASURE
            self._tiger_pos_1 = self.np_random.integers(0, 2)
        elif act_1 == 2: # Abrir Derecha
            total_reward += self.R_TIGER if self._tiger_pos_1 == 1 else self.R_TREASURE
            self._tiger_pos_1 = self.np_random.integers(0, 2)

        # --- Lógica del Problema 2 ---
        if act_2 == 0: # Escuchar
            total_reward += self.R_LISTEN
            obs_2 = (1 if self._tiger_pos_2 == 0 else 2) if self.np_random.random() < self.accuracy else (2 if self._tiger_pos_2 == 0 else 1)
        elif act_2 == 1: # Abrir Izquierda
            total_reward += self.R_TIGER if self._tiger_pos_2 == 0 else self.R_TREASURE
            self._tiger_pos_2 = self.np_random.integers(0, 2)
        elif act_2 == 2: # Abrir Derecha
            total_reward += self.R_TIGER if self._tiger_pos_2 == 1 else self.R_TREASURE
            self._tiger_pos_2 = self.np_random.integers(0, 2)

        # --- Finalizar el paso ---
        self._current_steps += 1
        terminated = False 
        truncated = self._current_steps >= self.max_episode_steps
        
        observation = np.array([obs_1, obs_2], dtype=np.int64)
        info = self._get_info()

        self._last_action = action
        self._last_observation = observation
        self._last_reward = total_reward
        
        if self._render_mode == "human":
            self.render()

        return observation, total_reward, terminated, truncated, info

    def render(self):
        if self._render_mode == "ansi":
            outfile = io.StringIO()
            outfile.write(f"Paso: {self._current_steps}\n")
            outfile.write(f"  Tigre 1: {'Izquierda' if self._tiger_pos_1 == 0 else 'Derecha'}\n")
            outfile.write(f"  Tigre 2: {'Izquierda' if self._tiger_pos_2 == 0 else 'Derecha'}\n")
            return outfile.getvalue()
        
        elif self._render_mode == "human":
            os.system('cls' if os.name == 'nt' else 'clear')
            print(f"====== DOS TIGRES (Paso: {self._current_steps}) ======")
            print(f"Precisión del Sonido: {self.accuracy * 100}%\n")
            pos1 = self._tiger_pos_1
            doors1_visual = "[ 🐯 Tigre | 💰 Tesoro ]" if pos1 == 0 else "[ 💰 Tesoro | 🐯 Tigre ]"
            print(f"Problema 1: {doors1_visual}")
            pos2 = self._tiger_pos_2
            doors2_visual = "[ 🐯 Tigre | 💰 Tesoro ]" if pos2 == 0 else "[ 💰 Tesoro | 🐯 Tigre ]"
            print(f"Problema 2: {doors2_visual}")
            print("\n" + "="*30)
            
            if self._last_action is not None:
                act_1 = self._last_action // 3 
                act_2 = self._last_action % 3
                act1_str = ["Escuchar", "Abrir Izq", "Abrir Der"][act_1]
                act2_str = ["Escuchar", "Abrir Izq", "Abrir Der"][act_2]
                print(f"Última Acción:  [{act1_str}, {act2_str}]")
            else:
                print("Última Acción:  [N/A (Inicio de episodio)]")

            if self._last_observation is not None:
                obs1_str = ["Silencio", "Sonido Izq", "Sonido Der"][self._last_observation[0]]
                obs2_str = ["Silencio", "Sonido Izq", "Sonido Der"][self._last_observation[1]]
                print(f"Observación:    [{obs1_str}, {obs2_str}]")
            else:
                print("Observación:    [N/A]")
            
            print(f"Recompensa:     {self._last_reward}\n")
            time.sleep(0.1)

    def close(self):
        pass


# ==============================================================================
# === 2. RED NEURONAL (PSR-DRQN) ===============================================
# ==============================================================================

# --- Constantes de la Red ---
NUM_OBS_CATEGORIES_1 = 3  # [0: Silencio, 1: Sonido Izq, 2: Sonido Der]
NUM_OBS_CATEGORIES_2 = 3
NUM_ACTIONS = 9
EMBED_DIM_OBS = 8         # Dimensión para embeber cada observación
EMBED_DIM_ACT = 8         # Dimensión para embeber la acción previa
RNN_HIDDEN_DIM = 64       # Tamaño del estado oculto (nuestro PSR 'ψ_h')

class PSR_DRQN_Network(nn.Module):
    """
    Red DRQN con cabezas auxiliares de PSR (predicción de obs/reward).
    """
    def __init__(self):
        super().__init__()
        
        # 1. Capas de Embedding
        self.embed_obs_1 = nn.Embedding(NUM_OBS_CATEGORIES_1, EMBED_DIM_OBS)
        self.embed_obs_2 = nn.Embedding(NUM_OBS_CATEGORIES_2, EMBED_DIM_OBS)
        self.embed_prev_action = nn.Embedding(NUM_ACTIONS, EMBED_DIM_ACT)
        
        rnn_input_dim = (EMBED_DIM_OBS * 2) + EMBED_DIM_ACT
        
        # 2. Cuerpo Recurrente (El codificador de historial)
        self.rnn = nn.GRU(rnn_input_dim, RNN_HIDDEN_DIM, batch_first=True)
        
        # 3. CABEZAS (Salidas)
        
        # Cabeza 1: Q-Value (Tarea principal)
        self.q_head = nn.Linear(RNN_HIDDEN_DIM, NUM_ACTIONS)
        
        # Cabeza 2: PSR - Predicción de Recompensa
        self.psr_reward_head = nn.Linear(RNN_HIDDEN_DIM, 1)
        
        # Cabeza 3: PSR - Predicción de Próxima Observación
        self.psr_obs_1_head = nn.Linear(RNN_HIDDEN_DIM, NUM_OBS_CATEGORIES_1)
        self.psr_obs_2_head = nn.Linear(RNN_HIDDEN_DIM, NUM_OBS_CATEGORIES_2)

    def forward(self, obs, prev_actions, hidden_state):
        # obs shape: (B, T, 2)
        # prev_actions shape: (B, T)
        # hidden_state shape: (1, B, RNN_HIDDEN_DIM)
        # donde B=batch_size, T=sequence_length
        
        # Embeber entradas
        emb_obs_1 = self.embed_obs_1(obs[:, :, 0])
        emb_obs_2 = self.embed_obs_2(obs[:, :, 1])
        emb_prev_act = self.embed_prev_action(prev_actions)
        
        # Concatenar para formar la entrada del RNN
        rnn_input = torch.cat([emb_obs_1, emb_obs_2, emb_prev_act], dim=2)
        
        # Procesar con el RNN
        rnn_output, new_hidden_state = self.rnn(rnn_input, hidden_state)
        
        # Pasar el *historial* (rnn_output) por las cabezas
        q_values = self.q_head(rnn_output)
        pred_rewards = self.psr_reward_head(rnn_output)
        pred_obs_1_logits = self.psr_obs_1_head(rnn_output)
        pred_obs_2_logits = self.psr_obs_2_head(rnn_output)
        
        return {
            "q_values": q_values,
            "pred_rewards": pred_rewards,
            "pred_obs_1_logits": pred_obs_1_logits,
            "pred_obs_2_logits": pred_obs_2_logits,
            "hidden_state": new_hidden_state
        }

# ==============================================================================
# === 3. BUFFER DE REPETICIÓN POR SECUENCIAS ===================================
# ==============================================================================

# Tupla para almacenar transiciones individuales
Transition = namedtuple('Transition', 
                        ('obs', 'action', 'reward', 'next_obs', 'done'))

class SequenceReplayBuffer:
    """
    Buffer que almacena transiciones y muestrea secuencias (trazas).
    """
    def __init__(self, capacity, sequence_length):
        self.capacity = capacity
        self.sequence_length = sequence_length
        self.buffer = deque(maxlen=capacity)
        # Pads
        self.obs_pad = np.zeros(2, dtype=np.int64) # obs [0, 0]
        self.action_pad = 0 # acción 0
        self.reward_pad = 0.0
        self.done_pad = True

    def push(self, obs, action, reward, next_obs, done):
        """Guarda una transición."""
        self.buffer.append(Transition(obs, action, reward, next_obs, done))

    def sample(self, batch_size):
        """Muestrea un batch de secuencias."""
        
        # Pre-alocar tensores para las secuencias del batch
        batch_obs_seq = torch.zeros((batch_size, self.sequence_length, 2), dtype=torch.long)
        batch_action_seq = torch.zeros((batch_size, self.sequence_length), dtype=torch.long)
        batch_reward_seq = torch.zeros((batch_size, self.sequence_length), dtype=torch.float)
        batch_next_obs_seq = torch.zeros((batch_size, self.sequence_length, 2), dtype=torch.long)
        batch_done_seq = torch.zeros((batch_size, self.sequence_length), dtype=torch.float)

        for i in range(batch_size):
            # Elegir un índice final aleatorio para la secuencia
            end_idx = random.randint(0, len(self.buffer) - 1)
            start_idx = max(0, end_idx - self.sequence_length + 1)
            
            # Revisar si hay un 'done' en medio de la secuencia
            # Si es así, empezamos la secuencia DESPUÉS del done
            for k in range(start_idx, end_idx):
                if self.buffer[k].done:
                    start_idx = k + 1
            
            # Rellenar la secuencia (de atrás hacia adelante)
            seq_idx = self.sequence_length - 1
            for k in range(end_idx, start_idx - 1, -1):
                trans = self.buffer[k]
                batch_obs_seq[i, seq_idx] = torch.tensor(trans.obs)
                batch_action_seq[i, seq_idx] = torch.tensor(trans.action)
                batch_reward_seq[i, seq_idx] = torch.tensor(trans.reward)
                batch_next_obs_seq[i, seq_idx] = torch.tensor(trans.next_obs)
                batch_done_seq[i, seq_idx] = torch.tensor(trans.done)
                seq_idx -= 1
            
            # Rellenar el resto con PADDING (si la secuencia es corta)
            while seq_idx >= 0:
                batch_obs_seq[i, seq_idx] = torch.tensor(self.obs_pad)
                batch_action_seq[i, seq_idx] = torch.tensor(self.action_pad)
                batch_reward_seq[i, seq_idx] = torch.tensor(self.reward_pad)
                batch_next_obs_seq[i, seq_idx] = torch.tensor(self.obs_pad)
                batch_done_seq[i, seq_idx] = torch.tensor(self.done_pad)
                seq_idx -= 1

        return (
            batch_obs_seq.to(device),
            batch_action_seq.to(device),
            batch_reward_seq.to(device),
            batch_next_obs_seq.to(device),
            batch_done_seq.to(device)
        )

    def __len__(self):
        return len(self.buffer)

# ==============================================================================
# === 4. CLASE DEL AGENTE (PSR-DRQN) ===========================================
# ==============================================================================

# --- Hiperparámetros ---
BUFFER_SIZE = 50000        # Tamaño del replay buffer
BATCH_SIZE = 32            # Tamaño del batch muestreado
SEQUENCE_LENGTH = 10       # Longitud de las secuencias a muestrear
GAMMA = 0.99               # Factor de descuento
EPS_START = 1.0            # Epsilon inicial (exploración)
EPS_END = 0.05             # Epsilon final
EPS_DECAY = 10000          # Pasos para decaer epsilon
LR = 1e-4                  # Tasa de aprendizaje
TARGET_UPDATE_FREQ = 1000  # Pasos para actualizar la red objetivo
BETA_REWARD = 0.3          # Peso de la pérdida PSR (recompensa)
BETA_OBS = 0.5             # Peso de la pérdida PSR (observación)

# Usar GPU si está disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")


class Agent:
    def __init__(self):
        # Inicializar redes
        self.main_network = PSR_DRQN_Network().to(device)
        self.target_network = PSR_DRQN_Network().to(device)
        self.target_network.load_state_dict(self.main_network.state_dict())
        self.target_network.eval() # Target net solo para inferencia

        self.optimizer = optim.Adam(self.main_network.parameters(), lr=LR)
        
        self.buffer = SequenceReplayBuffer(BUFFER_SIZE, SEQUENCE_LENGTH)
        
        # Estado interno del agente
        self.hidden_state = None
        self.prev_action = 0 # Acción de padding inicial
        self.steps_done = 0

        # Criterios de pérdida (los movemos aquí)
        self.loss_fn_dqn = nn.MSELoss()
        self.loss_fn_psr_reward = nn.MSELoss()
        self.loss_fn_psr_obs = nn.CrossEntropyLoss()

    def reset_hidden_state(self, batch_size=1):
        """Reinicia el estado oculto del agente (al inicio de un episodio)."""
        self.hidden_state = torch.zeros(1, batch_size, RNN_HIDDEN_DIM, device=device)
        self.prev_action = 0 # Reiniciar acción previa a padding

    def get_action(self, obs):
        """Elige una acción (epsilon-greedy) y actualiza el estado oculto."""
        
        # Calcular epsilon
        epsilon = EPS_END + (EPS_START - EPS_END) * \
                  np.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1

        if random.random() > epsilon:
            # Explotación: elegir la mejor acción
            with torch.no_grad():
                # Preparar entradas (Batch=1, Seq_Len=1)
                obs_tensor = torch.tensor(obs, device=device, dtype=torch.long).unsqueeze(0).unsqueeze(0) # (1, 1, 2)
                prev_act_tensor = torch.tensor([self.prev_action], device=device, dtype=torch.long).unsqueeze(0) # (1, 1)

                preds = self.main_network(obs_tensor, prev_act_tensor, self.hidden_state)
                
                q_values = preds["q_values"]
                self.hidden_state = preds["hidden_state"] # Actualizar estado oculto
                
                action = q_values.squeeze(0).argmax().item()
        else:
            # Exploración: elegir acción aleatoria
            # ¡IMPORTANTE! Aún debemos actualizar el estado oculto
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, device=device, dtype=torch.long).unsqueeze(0).unsqueeze(0)
                prev_act_tensor = torch.tensor([self.prev_action], device=device, dtype=torch.long).unsqueeze(0)
                preds = self.main_network(obs_tensor, prev_act_tensor, self.hidden_state)
                self.hidden_state = preds["hidden_state"] # Actualizar estado oculto
            
            action = env.action_space.sample()
        
        self.prev_action = action # Guardar acción actual para el *próximo* paso
        return action

    def learn(self):
        """
        Muestrea un batch de secuencias y realiza un paso de optimización.
        """
        if len(self.buffer) < BATCH_SIZE:
            return None # No hay suficientes muestras
        
        # 1. Muestrear secuencias
        obs_seq, action_seq, reward_seq, next_obs_seq, done_seq = \
            self.buffer.sample(BATCH_SIZE)

        # 2. Preparar acciones previas para el RNN
        # La entrada en t=0 usa una acción de padding (0)
        pad_action = torch.zeros((BATCH_SIZE, 1), dtype=torch.long, device=device)
        # Desplazamos las acciones: [a_t, a_{t+1}] -> [PAD, a_t]
        prev_action_seq = torch.cat([pad_action, action_seq[:, :-1]], dim=1)

        # 3. Forward pass en la RED PRINCIPAL
        # El estado oculto inicial para el batch es siempre cero
        initial_hidden = torch.zeros(1, BATCH_SIZE, RNN_HIDDEN_DIM, device=device)
        
        main_preds = self.main_network(obs_seq, prev_action_seq, initial_hidden)
        
        # Q-values predichos para las acciones tomadas
        # main_preds["q_values"] tiene shape (B, T, 9)
        # action_seq tiene shape (B, T)
        q_values_predicted = torch.gather(main_preds["q_values"], 2, action_seq.unsqueeze(-1)).squeeze(-1)
        
        # 4. Calcular el Target de Bellman (L_DQN)
        with torch.no_grad():
            # Necesitamos las acciones previas para la *siguiente* observación
            # La acción previa para o_{t+1} es a_t
            prev_action_seq_target = action_seq # Sin padding, usamos la acción actual
            
            # (Double-DQN)
            # 1. Encontrar la mejor acción usando la red *principal*
            next_preds_main = self.main_network(next_obs_seq, prev_action_seq_target, initial_hidden)
            best_next_actions = torch.argmax(next_preds_main["q_values"], dim=2).unsqueeze(-1)
            
            # 2. Obtener el Q-value de esa acción desde la red *target*
            next_preds_target = self.target_network(next_obs_seq, prev_action_seq_target, initial_hidden)
            q_next_target = torch.gather(next_preds_target["q_values"], 2, best_next_actions).squeeze(-1)

            # Calcular el target Y_t
            q_target = reward_seq + GAMMA * (1 - done_seq) * q_next_target

        # ---- PÉRDIDA 1: L_DQN ----
        loss_dqn = self.loss_fn_dqn(q_values_predicted, q_target)

        # 5. Calcular Pérdidas Auxiliares (L_PSR)
        
        # ---- PÉRDIDA 2: L_PSR (Recompensa) ----
        # predecir r_t usando h_t
        pred_rewards = main_preds["pred_rewards"].squeeze(-1)
        loss_psr_reward = self.loss_fn_psr_reward(pred_rewards, reward_seq)
        
        # ---- PÉRDIDA 3: L_PSR (Observación) ----
        # predecir o_{t+1} usando h_t
        B, T, _ = main_preds["pred_obs_1_logits"].shape
        
        # Convertir a (B*T, Num_Clases) para CrossEntropy
        pred_obs_1_logits = main_preds["pred_obs_1_logits"].view(B * T, NUM_OBS_CATEGORIES_1)
        pred_obs_2_logits = main_preds["pred_obs_2_logits"].view(B * T, NUM_OBS_CATEGORIES_2)
        
        # Targets: (B*T)
        true_next_obs_1 = next_obs_seq[:, :, 0].view(B * T)
        true_next_obs_2 = next_obs_seq[:, :, 1].view(B * T)

        loss_psr_obs_1 = self.loss_fn_psr_obs(pred_obs_1_logits, true_next_obs_1)
        loss_psr_obs_2 = self.loss_fn_psr_obs(pred_obs_2_logits, true_next_obs_2)
        loss_psr_obs = loss_psr_obs_1 + loss_psr_obs_2
        
        # 6. Calcular Pérdida Total
        loss_total = loss_dqn + (BETA_REWARD * loss_psr_reward) + (BETA_OBS * loss_psr_obs)
        
        # 7. Backpropagation
        self.optimizer.zero_grad()
        loss_total.backward()
        # Opcional: Gradiente clipping
        # torch.nn.utils.clip_grad_norm_(self.main_network.parameters(), 1.0)
        self.optimizer.step()

        # Devolver pérdidas para logging
        return {
            "total_loss": loss_total.item(),
            "dqn_loss": loss_dqn.item(),
            "psr_reward_loss": loss_psr_reward.item(),
            "psr_obs_loss": loss_psr_obs.item()
        }

    def update_target_network(self):
        """Copia los pesos de la red principal a la red objetivo."""
        self.target_network.load_state_dict(self.main_network.state_dict())


# ==============================================================================
# === 5. LOOP DE ENTRENAMIENTO =================================================
# ==============================================================================

if __name__ == "__main__":
    
    # --- Configuración ---
    NUM_EPISODES = 1000
    MAX_STEPS_PER_EPISODE = 200 # Coincide con env.max_episode_steps
    TRAIN_EVERY_N_STEPS = 4     # Frecuencia de entrenamiento
    
    # --- Inicialización ---
    env = TwoTigersEnv(max_episode_steps=MAX_STEPS_PER_EPISODE)
    agent = Agent()
    
    # --- Logging ---
    episode_rewards = []
    all_losses = {
        "total": [], "dqn": [], "psr_rew": [], "psr_obs": []
    }
    
    total_steps = 0
    start_time = time.time()

    print("Iniciando entrenamiento...")

    # --- Loop principal de episodios ---
    for i_episode in trange(NUM_EPISODES):
        
        obs, _ = env.reset()
        agent.reset_hidden_state() # ¡MUY IMPORTANTE!
        
        episode_reward = 0
        
        for t in range(MAX_STEPS_PER_EPISODE):
            
            # 1. Agente elige acción
            action = agent.get_action(obs)
            
            # 2. Entorno ejecuta acción
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # 3. Guardar en buffer
            agent.buffer.push(obs, action, reward, next_obs, done)
            
            # 4. Actualizar estado y recompensa
            obs = next_obs
            episode_reward += reward
            total_steps += 1
            
            # 5. Entrenar la red
            if total_steps % TRAIN_EVERY_N_STEPS == 0:
                losses = agent.learn()
                if losses:
                    all_losses["total"].append(losses["total_loss"])
                    all_losses["dqn"].append(losses["dqn_loss"])
                    all_losses["psr_rew"].append(losses["psr_reward_loss"])
                    all_losses["psr_obs"].append(losses["psr_obs_loss"])
            
            # 6. Actualizar red objetivo
            if total_steps % TARGET_UPDATE_FREQ == 0:
                agent.update_target_network()
                # print("  *** Red objetivo actualizada ***")
            
            if done:
                break
        
        # --- Fin de episodio ---
        episode_rewards.append(episode_reward)
        
        if (i_episode + 1) % 100 == 0:
            end_time = time.time()
            avg_reward = np.mean(episode_rewards[-100:])
            avg_loss = np.mean(all_losses["total"][-100:]) if all_losses["total"] else 0
            steps_per_sec = (total_steps / (end_time - start_time))
            
            print(f"Episodio {i_episode+1}/{NUM_EPISODES} | "
                  f"Recompensa Media (100 ep): {avg_reward:.2f} | "
                  f"Loss Total Media: {avg_loss:.3f} | "
                  f"Epsilon: {agent.steps_done:,.0f} | "
                  f"SPS: {steps_per_sec:,.0f}")
            start_time = time.time() # Resetear timer

    print("Entrenamiento finalizado.")

    # --- 6. Graficar Resultados ---
    
    # Función para suavizar la curva
    def moving_average(data, window_size):
        return np.convolve(data, np.ones(window_size), 'valid') / window_size

    window = 50
    smoothed_rewards = moving_average(episode_rewards, window)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(smoothed_rewards)
    plt.title(f"Recompensas por Episodio (Suavizado {window})")
    plt.xlabel("Episodio")
    plt.ylabel("Recompensa Media")

    plt.subplot(1, 2, 2)
    plt.plot(moving_average(all_losses["total"], window), label="Loss Total")
    plt.plot(moving_average(all_losses["dqn"], window), label="Loss DQN (Control)")
    plt.plot(moving_average(all_losses["psr_rew"], window), label="Loss PSR (Reward)")
    plt.plot(moving_average(all_losses["psr_obs"], window), label="Loss PSR (Obs)")
    plt.title(f"Pérdidas (Suavizado {window})")
    plt.xlabel("Paso de Entrenamiento")
    plt.ylabel("Pérdida")
    plt.legend()
    plt.tight_layout()
    plt.show()