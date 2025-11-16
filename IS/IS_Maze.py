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
import matplotlib.pyplot as plt
from tqdm import trange

# Optional: stable_baselines3 Monitor for standard RL logging (monitor.csv)
try:
    from stable_baselines3.common.monitor import Monitor
    _HAS_SB3_MONITOR = True
except Exception:
    Monitor = None
    _HAS_SB3_MONITOR = False

# ==============================================================================
# === 1. ENTORNO (Tu código de KeyDoorMazeEnv) =================================
# ==============================================================================

class KeyDoorMazeEnv(gym.Env):
    """
    Entorno de Laberinto con Llave y Puerta (PO-KeyDoor) - Diseño "Encrucijada"
    Compatible con Gymnasium.
    (Versión del script del usuario)
    """
    def __init__(self, height=15, width=19, max_episode_steps=250, render_mode="ansi"):
        assert height >= 7 and height % 2 == 1, "Altura debe ser impar >= 7"
        assert width >= 11 and width % 2 == 1, "Anchura debe ser impar >= 11"
        
        self.height = height
        self.width = width
        self.max_episode_steps = max_episode_steps
        self._render_mode = render_mode

        # --- Generar el Mapa ---
        self._grid = np.zeros((height, width), dtype=np.int8) # 0: Piso
        self._grid[0, :] = 1; self._grid[-1, :] = 1; self._grid[:, 0] = 1; self._grid[:, -1] = 1
        
        mid_col = width // 2
        self._grid[1:-1, mid_col] = 1

        start_row = height - 2
        self._start_pos = (start_row, mid_col)
        self._grid[start_row - 1 : start_row + 1, mid_col] = 0
        
        fork_row = start_row - 2
        self._grid[fork_row, 1:mid_col] = 0
        self._grid[fork_row, mid_col+1:-1] = 0

        # Zigzag Izquierdo
        for r in range(4, fork_row, 2):
            if ((r - 4) // 2) % 2 == 0:
                self._grid[r, 1:mid_col-3] = 1
                self._grid[r, mid_col-3] = 0; self._grid[r+1, mid_col-3] = 0
            else:
                self._grid[r, 3:mid_col] = 1
                self._grid[r, 2] = 0; self._grid[r+1, 2] = 0
        
        # Zigzag Derecho
        for r in range(4, fork_row, 2):
            if ((r - 4) // 2) % 2 == 0:
                self._grid[r, mid_col+3:width-1] = 1
                self._grid[r, mid_col+2] = 0; self._grid[r+1, mid_col+2] = 0
            else:
                self._grid[r, mid_col+1:width-3] = 1
                self._grid[r, width-3] = 0; self._grid[r+1, width-3] = 0
        
        # Vestíbulos
        for r in range(1, 4):
            for c in range(2, mid_col - 1): self._grid[r, c] = 0
        for r in range(1, 4):
            for c in range(mid_col + 2, width - 2): self._grid[r, c] = 0

        # --- Posiciones de Objetos ---
        self._key_red_pos = (fork_row, 2)
        self._key_blue_pos = (fork_row, width - 3)
        self._door_red_pos = (2, mid_col - 5)
        self._door_blue_pos_trap = (2, mid_col - 3)
        self._door_blue_pos = (2, mid_col + 3)
        self._door_red_pos_trap = (2, mid_col + 5)
        
        # --- Espacios de Gymnasium ---
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0, high=6, shape=(3, 3), dtype=np.int8
        )
        
        self._agent_loc = self._start_pos
        self._has_key = 0 # 0=no, 2=roja, 3=azul
        self._episode_steps = 0
        self._current_grid_view = self._grid.copy()

    def _get_obs(self):
        obs = np.ones((3, 3), dtype=np.int8)
        x, y = self._agent_loc
        for obs_i, grid_i in enumerate(range(x - 1, x + 2)):
            for obs_j, grid_j in enumerate(range(y - 1, y + 2)):
                if 0 <= grid_i < self.height and 0 <= grid_j < self.width:
                    obs[obs_i, obs_j] = self._current_grid_view[grid_i, grid_j]
        obs[1, 1] = 6 # 6 es el agente
        return obs
    
    def _get_info(self):
        return {
            "agent_location": self._agent_loc,
            "has_key": self._has_key, # 0, 2, o 3
            "steps": self._episode_steps
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._agent_loc = self._start_pos
        self._has_key = 0
        self._episode_steps = 0
        
        self._current_grid_view = self._grid.copy()
        self._current_grid_view[self._key_red_pos] = 2
        self._current_grid_view[self._key_blue_pos] = 3
        self._current_grid_view[self._door_red_pos] = 4
        self._current_grid_view[self._door_blue_pos] = 5
        self._current_grid_view[self._door_blue_pos_trap] = 5 
        self._current_grid_view[self._door_red_pos_trap] = 4 

        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info

    def step(self, action):
        _action_to_delta = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        delta = _action_to_delta[action]
        new_x, new_y = self._agent_loc[0] + delta[0], self._agent_loc[1] + delta[1]
        
        reward = -0.01 
        terminated = False
        
        if self._current_grid_view[new_x, new_y] == 1: # Choca con muro
            reward -= 0.1 
        else:
            self._agent_loc = (new_x, new_y)
            cell_value = self._current_grid_view[self._agent_loc]
            
            if cell_value == 2: # Recoge Llave Roja
                if self._has_key == 0:
                    self._has_key = 2
                    self._current_grid_view[self._key_red_pos] = 0
                    reward += 0.1
            
            elif cell_value == 3: # Recoge Llave Azul
                if self._has_key == 0:
                    self._has_key = 3
                    self._current_grid_view[self._key_blue_pos] = 0
                    reward += 0.1

            elif cell_value == 4: # Puerta Roja
                if self._has_key == 2:
                    reward += 10.0
                    terminated = True
                elif self._has_key == 3:
                    reward -= 5.0
                    terminated = True
            
            elif cell_value == 5: # Puerta Azul
                if self._has_key == 3:
                    reward += 10.0
                    terminated = True
                elif self._has_key == 2:
                    reward -= 5.0
                    terminated = True
        
        self._episode_steps += 1
        truncated = self._episode_steps >= self.max_episode_steps
        
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def render(self):
        # ... (La función de render es útil para depurar pero omitida aquí) ...
        pass
    
    def close(self):
        pass


# ==============================================================================
# === 2. RED NEURONAL (PSR-DRQN con CNN) =======================================
# ==============================================================================

# --- Constantes de la Red ---
# El espacio de obs es (3, 3)
OBS_SHAPE = (3, 3)
NUM_ACTIONS = 4
EMBED_DIM_ACT = 16        # Dimensión para embeber la acción previa
RNN_HIDDEN_DIM = 128      # Tamaño del estado oculto (nuestro PSR 'ψ_h')
NUM_KEY_CLASSES = 3       # 0=No, 1=Roja, 2=Azul

class PSR_DRQN_Network_Maze(nn.Module):
    """
    Red DRQN con cuerpo CNN y cabezas auxiliares de PSR.
    """
    def __init__(self):
        super().__init__()
        
        # 1. Cuerpo Convolucional (Encoder de Observación)
        # Recibe (B*T, 1, 3, 3)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=2, stride=1), # Salida (16, 2, 2)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=2, stride=1), # Salida (32, 1, 1)
            nn.ReLU(),
            nn.Flatten() # Salida (32)
        )
        cnn_feat_dim = 32
        
        # 2. Embedding de Acción
        self.embed_prev_action = nn.Embedding(NUM_ACTIONS, EMBED_DIM_ACT)
        
        # 3. Cuerpo Recurrente
        rnn_input_dim = cnn_feat_dim + EMBED_DIM_ACT
        self.rnn = nn.GRU(rnn_input_dim, RNN_HIDDEN_DIM, batch_first=True)
        
        # 4. CABEZAS (Salidas)
        
        # Cabeza 1: Q-Value
        self.q_head = nn.Linear(RNN_HIDDEN_DIM, NUM_ACTIONS)
        
        # Cabeza 2: PSR - Predicción de Recompensa
        self.psr_reward_head = nn.Linear(RNN_HIDDEN_DIM, 1)
        
        # Cabeza 3: PSR - Predicción de Estado de Llave
        # (0=No, 1=Roja, 2=Azul)
        self.psr_key_head = nn.Linear(RNN_HIDDEN_DIM, NUM_KEY_CLASSES)

    def forward(self, obs, prev_actions, hidden_state):
        # obs shape: (B, T, 3, 3)
        # prev_actions shape: (B, T)
        # hidden_state shape: (1, B, RNN_HIDDEN_DIM)
        
        B, T, H, W = obs.shape
        
        # 1. Procesar Observación con CNN
        # Convertir a (B*T, 1, H, W) para la CNN
        cnn_in = obs.view(B * T, 1, H, W)
        cnn_out = self.cnn(cnn_in) # (B*T, 32)
        # Devolver a formato secuencia (B, T, 32)
        cnn_out_seq = cnn_out.view(B, T, -1)
        
        # 2. Procesar Acción Previa
        emb_prev_act = self.embed_prev_action(prev_actions) # (B, T, 16)
        
        # 3. Concatenar y pasar por RNN
        rnn_input = torch.cat([cnn_out_seq, emb_prev_act], dim=2)
        
        rnn_output, new_hidden_state = self.rnn(rnn_input, hidden_state)
        
        # 4. Pasar por las cabezas
        q_values = self.q_head(rnn_output)
        pred_rewards = self.psr_reward_head(rnn_output)
        pred_key_logits = self.psr_key_head(rnn_output)
        
        return {
            "q_values": q_values,
            "pred_rewards": pred_rewards,
            "pred_key_logits": pred_key_logits,
            "hidden_state": new_hidden_state
        }

# ==============================================================================
# === 3. BUFFER DE REPETICIÓN (MODIFICADO) =====================================
# ==============================================================================

# Tupla para almacenar transiciones individuales (con 'key_state')
Transition = namedtuple('Transition', 
                        ('obs', 'action', 'reward', 'next_obs', 'done', 'key_state'))

# Mapa para convertir 0,2,3 -> 0,1,2
KEY_STATE_MAP = {0: 0, 2: 1, 3: 2}

class SequenceReplayBuffer:
    def __init__(self, capacity, sequence_length):
        self.capacity = capacity
        self.sequence_length = sequence_length
        self.buffer = deque(maxlen=capacity)
        
        # Pads
        self.obs_pad = np.zeros(OBS_SHAPE, dtype=np.int8)
        self.action_pad = 0
        self.reward_pad = 0.0
        self.done_pad = True
        self.key_state_pad = 0 # 0 = "Sin Llave"

    def push(self, obs, action, reward, next_obs, done, key_state):
        """Guarda una transición, mapeando el estado de la llave."""
        # Mapear [0, 2, 3] a [0, 1, 2] para la CrossEntropyLoss
        key_class = KEY_STATE_MAP.get(key_state, 0)
        self.buffer.append(Transition(obs, action, reward, next_obs, done, key_class))

    def sample(self, batch_size):
        batch_obs_seq = torch.zeros((batch_size, self.sequence_length, *OBS_SHAPE), dtype=torch.int8)
        batch_action_seq = torch.zeros((batch_size, self.sequence_length), dtype=torch.long)
        batch_reward_seq = torch.zeros((batch_size, self.sequence_length), dtype=torch.float)
        batch_next_obs_seq = torch.zeros((batch_size, self.sequence_length, *OBS_SHAPE), dtype=torch.int8)
        batch_done_seq = torch.zeros((batch_size, self.sequence_length), dtype=torch.float)
        batch_key_seq = torch.zeros((batch_size, self.sequence_length), dtype=torch.long) # ¡NUEVO!

        for i in range(batch_size):
            end_idx = random.randint(0, len(self.buffer) - 1)
            start_idx = max(0, end_idx - self.sequence_length + 1)
            
            for k in range(start_idx, end_idx):
                if self.buffer[k].done:
                    start_idx = k + 1
            
            seq_idx = self.sequence_length - 1
            for k in range(end_idx, start_idx - 1, -1):
                trans = self.buffer[k]
                batch_obs_seq[i, seq_idx] = torch.tensor(trans.obs)
                batch_action_seq[i, seq_idx] = torch.tensor(trans.action)
                batch_reward_seq[i, seq_idx] = torch.tensor(trans.reward)
                batch_next_obs_seq[i, seq_idx] = torch.tensor(trans.next_obs)
                batch_done_seq[i, seq_idx] = torch.tensor(trans.done)
                batch_key_seq[i, seq_idx] = torch.tensor(trans.key_state) # ¡NUEVO!
                seq_idx -= 1
            
            while seq_idx >= 0:
                batch_obs_seq[i, seq_idx] = torch.tensor(self.obs_pad)
                batch_action_seq[i, seq_idx] = torch.tensor(self.action_pad)
                batch_reward_seq[i, seq_idx] = torch.tensor(self.reward_pad)
                batch_next_obs_seq[i, seq_idx] = torch.tensor(self.obs_pad)
                batch_done_seq[i, seq_idx] = torch.tensor(self.done_pad)
                batch_key_seq[i, seq_idx] = torch.tensor(self.key_state_pad) # ¡NUEVO!
                seq_idx -= 1

        return (
            batch_obs_seq.to(device),
            batch_action_seq.to(device),
            batch_reward_seq.to(device),
            batch_next_obs_seq.to(device),
            batch_done_seq.to(device),
            batch_key_seq.to(device) # ¡NUEVO!
        )

    def __len__(self):
        return len(self.buffer)

# ==============================================================================
# === 4. CLASE DEL AGENTE (PSR-DRQN) ===========================================
# ==============================================================================

# --- Hiperparámetros ---
BUFFER_SIZE = 50000
BATCH_SIZE = 32
SEQUENCE_LENGTH = 16       # Aumentado para problemas de memoria más largos
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 20000
LR = 5e-5                  # Tasa de aprendizaje más baja para CNN/RNN
TARGET_UPDATE_FREQ = 1000
BETA_REWARD = 0.2          # Peso de la pérdida PSR (recompensa)
BETA_KEY = 0.8             # Peso de la pérdida PSR (estado de llave) - ¡Importante!

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

class Agent:
    def __init__(self):
        self.main_network = PSR_DRQN_Network_Maze().to(device)
        self.target_network = PSR_DRQN_Network_Maze().to(device)
        self.target_network.load_state_dict(self.main_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.main_network.parameters(), lr=LR)
        self.buffer = SequenceReplayBuffer(BUFFER_SIZE, SEQUENCE_LENGTH)
        
        self.hidden_state = None
        self.prev_action = 0
        self.steps_done = 0

        self.loss_fn_dqn = nn.MSELoss()
        self.loss_fn_psr_reward = nn.MSELoss()
        self.loss_fn_psr_key = nn.CrossEntropyLoss() # Para clasificación de llaves

    def reset_hidden_state(self, batch_size=1):
        self.hidden_state = torch.zeros(1, batch_size, RNN_HIDDEN_DIM, device=device)
        self.prev_action = 0

    def get_action(self, obs):
        epsilon = EPS_END + (EPS_START - EPS_END) * \
                  np.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1

        if random.random() > epsilon:
            with torch.no_grad():
                # Convertir obs a float para la CNN
                obs_tensor = torch.tensor(obs, device=device, dtype=torch.float).unsqueeze(0).unsqueeze(0) # (1, 1, 3, 3)
                prev_act_tensor = torch.tensor([self.prev_action], device=device, dtype=torch.long).unsqueeze(0) # (1, 1)

                preds = self.main_network(obs_tensor, prev_act_tensor, self.hidden_state)
                q_values = preds["q_values"]
                self.hidden_state = preds["hidden_state"]
                action = q_values.squeeze(0).argmax().item()
        else:
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, device=device, dtype=torch.float).unsqueeze(0).unsqueeze(0)
                prev_act_tensor = torch.tensor([self.prev_action], device=device, dtype=torch.long).unsqueeze(0)
                preds = self.main_network(obs_tensor, prev_act_tensor, self.hidden_state)
                self.hidden_state = preds["hidden_state"]
            action = env.action_space.sample()
        
        self.prev_action = action
        return action

    def learn(self):
        if len(self.buffer) < BATCH_SIZE:
            return None
        
        obs_seq, action_seq, reward_seq, next_obs_seq, done_seq, key_seq = \
            self.buffer.sample(BATCH_SIZE)

        # Convertir obs a float para la CNN
        obs_seq_float = obs_seq.float()
        next_obs_seq_float = next_obs_seq.float()

        # Acciones previas
        pad_action = torch.zeros((BATCH_SIZE, 1), dtype=torch.long, device=device)
        prev_action_seq = torch.cat([pad_action, action_seq[:, :-1]], dim=1)

        # Estado oculto inicial
        initial_hidden = torch.zeros(1, BATCH_SIZE, RNN_HIDDEN_DIM, device=device)
        
        # 1. Forward pass RED PRINCIPAL
        main_preds = self.main_network(obs_seq_float, prev_action_seq, initial_hidden)
        q_values_predicted = torch.gather(main_preds["q_values"], 2, action_seq.unsqueeze(-1)).squeeze(-1)
        
        # 2. Calcular Target de Bellman (L_DQN)
        with torch.no_grad():
            prev_action_seq_target = action_seq
            
            # Double-DQN
            next_preds_main = self.main_network(next_obs_seq_float, prev_action_seq_target, initial_hidden)
            best_next_actions = torch.argmax(next_preds_main["q_values"], dim=2).unsqueeze(-1)
            
            next_preds_target = self.target_network(next_obs_seq_float, prev_action_seq_target, initial_hidden)
            q_next_target = torch.gather(next_preds_target["q_values"], 2, best_next_actions).squeeze(-1)

            q_target = reward_seq + GAMMA * (1 - done_seq) * q_next_target

        # ---- PÉRDIDA 1: L_DQN ----
        loss_dqn = self.loss_fn_dqn(q_values_predicted, q_target)

        # 3. Calcular Pérdidas Auxiliares (L_PSR)
        B, T, _ = main_preds["pred_key_logits"].shape
        
        # ---- PÉRDIDA 2: L_PSR (Recompensa) ----
        pred_rewards = main_preds["pred_rewards"].squeeze(-1) # (B, T)
        loss_psr_reward = self.loss_fn_psr_reward(pred_rewards, reward_seq)
        
        # ---- PÉRDIDA 3: L_PSR (Estado de Llave) ----
        # predecir P(key_state | h_t)
        
        # (B, T, 3) -> (B*T, 3)
        pred_key_logits = main_preds["pred_key_logits"].view(B * T, NUM_KEY_CLASSES)
        # (B, T) -> (B*T)
        true_key_states = key_seq.view(B * T)

        loss_psr_key = self.loss_fn_psr_key(pred_key_logits, true_key_states)
        
        # 4. Calcular Pérdida Total
        loss_total = loss_dqn + (BETA_REWARD * loss_psr_reward) + (BETA_KEY * loss_psr_key)
        
        # 5. Backpropagation
        self.optimizer.zero_grad()
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(self.main_network.parameters(), 1.0)
        self.optimizer.step()

        return {
            "total_loss": loss_total.item(),
            "dqn_loss": loss_dqn.item(),
            "psr_reward_loss": loss_psr_reward.item(),
            "psr_key_loss": loss_psr_key.item() # Renombrada
        }

    def update_target_network(self):
        self.target_network.load_state_dict(self.main_network.state_dict())

# ==============================================================================
# === 5. LOOP DE ENTRENAMIENTO =================================================
# ==============================================================================

if __name__ == "__main__":
    
    NUM_EPISODES = 1000
    MAX_STEPS_PER_EPISODE = 1000
    TRAIN_EVERY_N_STEPS = 4
    
    # Create environment. If stable_baselines3 Monitor is available, wrap env to
    # save monitor.csv logs to logs/IS_Maze/monitor_<timestamp>.csv
    log_dir = os.path.join("logs", "IS_Maze")
    if _HAS_SB3_MONITOR:
        os.makedirs(log_dir, exist_ok=True)
        monitor_file = os.path.join(log_dir, f"monitor_{int(time.time())}.csv")
        env = Monitor(KeyDoorMazeEnv(max_episode_steps=MAX_STEPS_PER_EPISODE), monitor_file)
        print(f"Monitor habilitado. Logs -> {monitor_file}")
    else:
        env = KeyDoorMazeEnv(max_episode_steps=MAX_STEPS_PER_EPISODE)
        print("stable_baselines3 Monitor no disponible. Instala 'stable-baselines3' para guardar logs con Monitor.")

    agent = Agent()
    
    episode_rewards = []
    episode_lengths = []
    all_losses = {
        "total": [], "dqn": [], "psr_rew": [], "psr_key": []
    }
    
    total_steps = 0
    start_time = time.time()

    print("Iniciando entrenamiento en KeyDoorMazeEnv...")

    for i_episode in trange(NUM_EPISODES):
        obs, info = env.reset()
        agent.reset_hidden_state()
        
        episode_reward = 0
        episode_steps = 0
        
        for t in range(MAX_STEPS_PER_EPISODE):
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # ¡IMPORTANTE! Guardar el estado de la llave del info
            agent.buffer.push(obs, action, reward, next_obs, done, info["has_key"])
            
            obs = next_obs
            episode_reward += reward
            episode_steps += 1
            total_steps += 1
            
            if total_steps % TRAIN_EVERY_N_STEPS == 0:
                losses = agent.learn()
                if losses:
                    all_losses["total"].append(losses["total_loss"])
                    all_losses["dqn"].append(losses["dqn_loss"])
                    all_losses["psr_rew"].append(losses["psr_reward_loss"])
                    all_losses["psr_key"].append(losses["psr_key_loss"])
            
            if total_steps % TARGET_UPDATE_FREQ == 0:
                agent.update_target_network()
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_steps)
        
        if (i_episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_length = np.mean(episode_lengths[-100:])
            avg_loss_key = np.mean(all_losses["psr_key"][-100:]) if all_losses["psr_key"] else 0
            steps_per_sec = int((total_steps) / (time.time() - start_time))
            
            print(f"Episodio {i_episode+1}/{NUM_EPISODES} | "
                  f"Recompensa Media (100 ep): {avg_reward:.2f} | "
                  f"Largo Promedio (100 ep): {avg_length:.1f} | "
                  f"Loss Llave (PSR): {avg_loss_key:.3f} | "
                  f"SPS: {steps_per_sec}")
            start_time = time.time()

    print("Entrenamiento finalizado.")

    # --- 6. Graficar Resultados ---
    def moving_average(data, window_size):
        return np.convolve(data, np.ones(window_size), 'valid') / window_size

    window = 50
    smoothed_rewards = moving_average(episode_rewards, window)
    smoothed_lengths = moving_average(episode_lengths, window)
    
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(smoothed_rewards)
    plt.title(f"Recompensas por Episodio (Suavizado {window})")
    plt.xlabel(f"Episodio (x{window})")
    plt.ylabel("Recompensa Media")

    plt.subplot(1, 3, 2)
    plt.plot(smoothed_lengths)
    plt.title(f"Largo Promedio de Episodios (Suavizado {window})")
    plt.xlabel(f"Episodio (x{window})")
    plt.ylabel("Pasos por Episodio")

    plt.subplot(1, 3, 3)
    plt.plot(moving_average(all_losses["total"], window), label="Loss Total")
    plt.plot(moving_average(all_losses["dqn"], window), label="Loss DQN (Control)")
    plt.plot(moving_average(all_losses["psr_key"], window), label="Loss PSR (Llave)", linestyle='--')
    plt.plot(moving_average(all_losses["psr_rew"], window), label="Loss PSR (Reward)", linestyle=':')
    plt.title(f"Pérdidas (Suavizado {window})")
    plt.xlabel(f"Paso de Entrenamiento (x{window})")
    plt.ylabel("Pérdida")
    plt.legend()
    plt.tight_layout()
    plt.show()