import gymnasium as gym
import numpy as np
import pandas as pd
import time
import os
import glob
import matplotlib.pyplot as plt
import random
from collections import deque
import io

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# --- 1. IMPORTAR ENTORNOS PERSONALIZADOS ---
# (Asegúrate de que este archivo esté en la carpeta correcta para importar Entornos)
try:
    from Entornos.TwoTigersEnv import TwoTigersEnv
    # (Descomenta los otros si los vas a usar)
    # from Entornos.PODoorEnv import POKeyDoorEnv
    # from Entornos.KeyDoorMazeEnv import KeyDoorMazeEnv
    # from Entornos.DelayedObsEnv import DelayedObsEnv
except ImportError:
    print("Advertencia: No se pudieron importar los entornos desde la carpeta 'Entornos'.")
    print("Asegúrate de que TwoTigersEnv.py esté en Entornos/")
    exit()


#########################################################################
#                                                                       #
# --- (INICIO) SECCIÓN 1: EL WRAPPER (PSR + ENTROPÍA) ---                 #
#                                                                       #
#########################################################################

class BeliefStateWrapper(gym.Wrapper):
    """
    Wrapper que transforma el POMDP TwoTigersEnv en un MDP sobre el 
    belief state (PSR) enriquecido con entropía.
    
    Estado de salida (Observación):
    Vector de 5 dimensiones: [b(s1), b(s2), b(s3), b(s4), Entropía(b)]
    """

    def __init__(self, env):
        super().__init__(env)
        
        # --- 1. Constantes del POMDP ---
        self.num_states = 4  # |S| = 4  [(SL,SL), (SL,SR), (SR,SL), (SR,SR)]
        self.num_actions = 9 # |A| = 9  [(AE,AE), (AE,AL), ..., (AR,AR)]
        self.num_obs = 9     # |O| = 3x3 = 9
        
        # --- 2. Construir Modelos (T y Z) ---
        self.T = self._build_transition_model() # Matriz [s, a, s'] (4x9x4)
        self.Z = self._build_observation_model()  # Matriz [s', a, o] (4x9x9)
        
        # --- 3. Belief Inicial (b0) ---
        self.b0 = np.full(self.num_states, 1.0 / self.num_states, dtype=np.float32)
        self.current_belief = self.b0.copy()

        # --- 4. Sobrescribir Espacio de Observación ---
        # 5 dimensiones: [b1, b2, b3, b4, entropia]
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=2.0, # La entropía max (log2(4)) es 2.0
            shape=(self.num_states + 1,), # Shape es 5
            dtype=np.float32
        )

    def _calculate_entropy(self, belief):
        """Calcula la entropía de Shannon del vector de belief."""
        # Añadimos 1e-9 para evitar log(0)
        belief_clipped = np.clip(belief, 1e-9, 1.0)
        entropy = -np.sum(belief_clipped * np.log2(belief_clipped))
        return np.float32(entropy)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        self.current_belief = self.b0.copy()
        current_entropy = self._calculate_entropy(self.current_belief)
        
        # Concatena el belief y la entropía
        obs_with_entropy = np.concatenate([self.current_belief, [current_entropy]])
        
        return obs_with_entropy, info

    def step(self, action):
        # 1. Ejecutar paso en el entorno real
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # 2. Mapear obs a índice (0-8)
        o_idx = self._map_obs_to_index(obs)
        
        # 3. Actualizar belief state
        b_tplus1 = self._update_belief(self.current_belief, action, o_idx)
        
        # 4. Guardar nuevo belief
        self.current_belief = b_tplus1
        
        # 5. Calcular entropía y crear estado enriquecido
        current_entropy = self._calculate_entropy(self.current_belief)
        obs_with_entropy = np.concatenate([self.current_belief, [current_entropy]])
        
        # 6. Devolver el nuevo estado enriquecido
        return obs_with_entropy, reward, terminated, truncated, info

    # --- Motor del Filtro Bayesiano ---

    def _update_belief(self, b_t, a_idx, o_idx):
        """
        Actualiza el belief state usando el filtro Bayesiano.
        """
        # --- 1. Predicción ---
        T_a = self.T[:, a_idx, :]  # Matriz T(s, s' | a) de forma (4, 4)
        b_hat = T_a.T @ b_t        # Vector de forma (4,)
        
        # --- 2. Actualización ---
        Z_o = self.Z[:, a_idx, o_idx] # Vector Z(o | s', a) de forma (4,)
        b_new_unnormalized = Z_o * b_hat
        
        # --- 3. Normalización ---
        prob_obs = np.sum(b_new_unnormalized)
        
        if prob_obs < 1e-9:
            return self.b0.copy() # Fallback
        else:
            return b_new_unnormalized / prob_obs

    def _map_obs_to_index(self, obs):
        """
        Convierte la observación [o1, o2] (donde o_i in {0,1,2}) 
        a un índice único (0-8) usando base-3.
        """
        if isinstance(obs, (list, np.ndarray)) and len(obs) == 2:
            o1, o2 = obs
            o_idx = int(o1) * 3 + int(o2)
            if 0 <= o_idx < self.num_obs:
                return o_idx
        
        raise ValueError(f"Formato de observación no reconocida: {obs}.")

    # --- Funciones de Construcción de Modelos (Helpers) ---
    
    def _build_observation_model(self):
        """
        ¡CORREGIDO (v3)!
        Construye el tensor Z(s', a, o) = P(o | s', a) de 4x9x9.
        Alineado con el mapeo del entorno: {0: NO_OBS, 1: OL, 2: OR}
        """
        map_s_i = {'SL': 0, 'SR': 1}
        map_a_i = {'AE': 0, 'AL': 1, 'AR': 2}
        map_o_i = {'NO_OBS': 0, 'OL': 1, 'OR': 2}
        map_s = {0: ('SL', 'SL'), 1: ('SL', 'SR'), 2: ('SR', 'SL'), 3: ('SR', 'SR')}
        map_a = {
            0: ('AE', 'AE'), 1: ('AE', 'AL'), 2: ('AE', 'AR'),
            3: ('AL', 'AE'), 4: ('AR', 'AE'), 5: ('AL', 'AL'),
            6: ('AL', 'AR'), 7: ('AR', 'AL'), 8: ('AR', 'AR')
        }
        map_o = {
            0: ('NO_OBS', 'NO_OBS'), 1: ('NO_OBS', 'OL'), 2: ('NO_OBS', 'OR'),
            3: ('OL', 'NO_OBS'), 4: ('OL', 'OL'), 5: ('OL', 'OR'),
            6: ('OR', 'NO_OBS'), 7: ('OR', 'OL'), 8: ('OR', 'OR')
        }

        # 1. Tensor Z_i base (2x3x3) -> Z_i[s_i, a_i, o_i]
        Z_i = np.zeros((2, 3, 3))
        Z_i[0, 0, :] = [0.0, 0.85, 0.15]  # P(o|s=SL, a=AE)
        Z_i[1, 0, :] = [0.0, 0.15, 0.85]  # P(o|s=SR, a=AE)
        Z_i[:, 1, :] = [1.0, 0.0, 0.0]    # P(o|s=*, a=AL)
        Z_i[:, 2, :] = [1.0, 0.0, 0.0]    # P(o|s=*, a=AR)

        # 2. Tensor Z final (4x9x9) -> Z[s', a, o]
        Z = np.zeros((self.num_states, self.num_actions, self.num_obs))
        for s_prime_idx in range(self.num_states):
            for a_idx in range(self.num_actions):
                for o_idx in range(self.num_obs):
                    s1_str, s2_str = map_s[s_prime_idx]
                    a1_str, a2_str = map_a[a_idx]
                    o1_str, o2_str = map_o[o_idx]
                    
                    prob_z1 = Z_i[map_s_i[s1_str], map_a_i[a1_str], map_o_i[o1_str]]
                    prob_z2 = Z_i[map_s_i[s2_str], map_a_i[a2_str], map_o_i[o2_str]]
                    
                    Z[s_prime_idx, a_idx, o_idx] = prob_z1 * prob_z2
        return Z

    def _build_transition_model(self):
        """
        Construye el tensor T(s, a, s') = P(s' | s, a) de 4x9x4.
        """
        map_s_i = {'SL': 0, 'SR': 1}
        map_a_i = {'AE': 0, 'AL': 1, 'AR': 2}
        map_s = {0: ('SL', 'SL'), 1: ('SL', 'SR'), 2: ('SR', 'SL'), 3: ('SR', 'SR')}
        map_a = {
            0: ('AE', 'AE'), 1: ('AE', 'AL'), 2: ('AE', 'AR'),
            3: ('AL', 'AE'), 4: ('AR', 'AE'), 5: ('AL', 'AL'),
            6: ('AL', 'AR'), 7: ('AR', 'AL'), 8: ('AR', 'AR')
        }

        # 1. Tensor T_i base (2x3x2) -> T_i[s, a, s']
        T_i = np.zeros((2, 3, 2))
        T_i[0, 0, :] = [1.0, 0.0] # (s=SL, a=AE) -> s'=SL
        T_i[1, 0, :] = [0.0, 1.0] # (s=SR, a=AE) -> s'=SR
        T_i[:, 1, :] = 0.5         # (s=*, a=AL) -> s' ~ Unif(0.5, 0.5)
        T_i[:, 2, :] = 0.5         # (s=*, a=AR) -> s' ~ Unif(0.5, 0.5)

        # 2. Tensor T final (4x9x4) -> T[s, a, s']
        T = np.zeros((self.num_states, self.num_actions, self.num_states))
        for s_idx in range(self.num_states):
            for a_idx in range(self.num_actions):
                for s_prime_idx in range(self.num_states):
                    s1_str, s2_str = map_s[s_idx]
                    a1_str, a2_str = map_a[a_idx]
                    s1_p_str, s2_p_str = map_s[s_prime_idx]
                    
                    prob_t1 = T_i[map_s_i[s1_str], map_a_i[a1_str], map_s_i[s1_p_str]]
                    prob_t2 = T_i[map_s_i[s2_str], map_a_i[a2_str], map_s_i[s2_p_str]]
                    
                    T[s_idx, a_idx, s_prime_idx] = prob_t1 * prob_t2
        return T

#########################################################################
#                                                                       #
# --- (INICIO) SECCIÓN 2: AGENTE DQN CUSTOM (CON PER) ---                 #
#                                                                       #
#########################################################################

class QNet(nn.Module):
    """La red neuronal que estima los Q-values."""
    def __init__(self, state_space: int, action_space: int, net_arch: list = [64, 64]):
        super(QNet, self).__init__()
        self.state_space = state_space
        self.action_space = action_space
        
        layers = []
        input_dim = state_space
        for layer_size in net_arch:
            layers.append(nn.Linear(input_dim, layer_size))
            layers.append(nn.ReLU())
            input_dim = layer_size
        layers.append(nn.Linear(input_dim, action_space))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    def sample_action(self, obs, epsilon: float):
        """Epsilon-greedy action selection"""
        if random.random() < epsilon:
            return random.randint(0, self.action_space - 1)
        else:
            with torch.no_grad():
                return self.forward(obs).argmax().item()

# -----------------------------
# Prioritized Replay Buffer
# -----------------------------
class PrioritizedReplayBuffer:
    """Buffer de repetición priorizado (PER) basado en numpy."""
    
    def __init__(self, obs_dim: int, size: int, batch_size: int = 32, alpha: float = 0.6):
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size], dtype=np.float32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        
        self.priorities = np.zeros(size, dtype=np.float32)
        self.epsilon = 1e-6  # Evitar prioridad 0
        self.alpha = alpha   # [0..1] 0=uniforme, 1=totalmente priorizado
        
        self.max_size = size
        self.batch_size = batch_size
        self.ptr = 0
        self.size = 0

    def put(self, obs, act, rew, next_obs, done):
        # Asignar prioridad máxima a nuevas transiciones
        max_prio = np.max(self.priorities) if self.size > 0 else 1.0
        self.priorities[self.ptr] = max_prio

        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, beta: float = 0.4) -> dict[str, np.ndarray]:
        """Muestrea del buffer usando prioridades."""
        if self.size == 0:
            return None, None, None

        # 1. Obtener probabilidades
        prios = self.priorities[:self.size]
        probs = prios ** self.alpha
        probs /= probs.sum()

        # 2. Muestrear índices
        idxs = np.random.choice(self.size, self.batch_size, p=probs, replace=True)

        # 3. Obtener muestras
        samples = dict(
            obs=self.obs_buf[idxs],
            next_obs=self.next_obs_buf[idxs],
            acts=self.acts_buf[idxs],
            rews=self.rews_buf[idxs],
            done=self.done_buf[idxs]
        )

        # 4. Calcular pesos de Importance Sampling (IS)
        total = self.size
        weights = (total * probs[idxs]) ** (-beta)
        weights /= weights.max() # Normalizar
        weights = np.array(weights, dtype=np.float32)

        return samples, idxs, weights

    def update_priorities(self, batch_indices, batch_priorities):
        """Actualiza las prioridades de las transiciones muestreadas."""
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = (prio + self.epsilon)

    def __len__(self):
        return self.size

# -----------------------------
# DQN Agent (Encapsulates training)
# -----------------------------
class DQNAgent:
    def __init__(self, 
                 state_space: int, 
                 action_space: int, 
                 h_params: dict, # Diccionario de hiperparámetros
                 device='cpu'
                ):
        
        self.device = device
        self.gamma = h_params['gamma']
        self.batch_size = h_params['batch_size']
        self.tau = h_params['tau']

        # Q-networks
        net_arch = h_params.get('policy_kwargs', {}).get('net_arch', [64, 64])
        self.q_net = QNet(state_space, action_space, net_arch).to(device)
        self.target_q_net = QNet(state_space, action_space, net_arch).to(device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=h_params['learning_rate'])

        # Replay buffer PER
        self.replay_buffer = PrioritizedReplayBuffer(
            obs_dim=state_space, 
            size=h_params['buffer_size'], 
            batch_size=h_params['batch_size'],
            alpha=h_params['alpha']
        )
        
        # Hiperparámetros PER
        self.beta_start = h_params['beta_start']
        self.beta = self.beta_start
        self.beta_increment = (1.0 - self.beta_start) / h_params['beta_frames']
        self.total_steps = 0 
        
    def store_transition(self, obs, act, rew, next_obs, done):
        """Método de ayuda para poblar el buffer"""
        self.replay_buffer.put(obs, act, rew, next_obs, done)

    def update(self):
        """Entrena Q-network desde el PER. Devuelve la pérdida."""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Anneal beta y muestrear del PER
        self.beta = min(1.0, self.beta_start + self.total_steps * self.beta_increment)
        self.total_steps += 1
        
        samples, idxs, weights = self.replay_buffer.sample(self.beta)
        
        states = torch.FloatTensor(samples['obs']).to(self.device)
        actions = torch.LongTensor(samples['acts'].reshape(-1,1)).to(self.device)
        rewards = torch.FloatTensor(samples['rews'].reshape(-1,1)).to(self.device)
        next_states = torch.FloatTensor(samples['next_obs']).to(self.device)
        dones = torch.FloatTensor(samples['done'].reshape(-1,1)).to(self.device)
        weights_tensor = torch.FloatTensor(weights).reshape(-1, 1).to(self.device)

        # Compute target
        q_target_max = self.target_q_net(next_states).max(1)[0].unsqueeze(1).detach()
        targets = rewards + self.gamma * q_target_max * (1 - dones)

        q_out = self.q_net(states)
        q_a = q_out.gather(1, actions)

        # Calcular pérdida ponderada y errores TD
        elementwise_loss = F.smooth_l1_loss(q_a, targets, reduction='none')
        loss = (elementwise_loss * weights_tensor).mean()
        td_errors = (targets - q_a).abs().detach().cpu().numpy().squeeze()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Actualizar las prioridades en el buffer
        self.replay_buffer.update_priorities(idxs, td_errors)
        
        return loss.item()

    def soft_update(self):
        """Soft update target network (usa self.tau)"""
        for target_param, local_param in zip(self.target_q_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0 - self.tau)*target_param.data)

    def save_model(self, path='dqn_model.pth'):
        torch.save(self.q_net.state_dict(), path)

    def load_model(self, path):
        self.q_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_q_net.load_state_dict(self.q_net.state_dict())

#########################################################################
#                                                                       #
# --- (INICIO) SECCIÓN 3: ARNÉS DE EXPERIMENTACIÓN ---                    #
#                                                                       #
#########################################################################

def check_gpu():
    """Verifica si PyTorch puede detectar y usar la GPU."""
    print(f"\n{'='*40}")
    print(" Verificando disponibilidad de GPU (PyTorch)...")
    print(f"{'='*40}")
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        current_dev_idx = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_dev_idx)
        print(f"✅ ¡Éxito! GPU disponible.")
        print(f"   Usando dispositivo {current_dev_idx}: {device_name}")
        return torch.device("cuda")
    else:
        print(f"❌ GPU no disponible. Usando CPU.")
        return torch.device("cpu")
    print(f"{'='*40}\n")
    

class ExperimentLogger:
    """Clase simple para guardar logs en formato monitor.csv."""
    def __init__(self, log_dir):
        self.log_file_path = os.path.join(log_dir, "monitor.csv")
        self.start_time = time.time()
        
        # Escribir el encabezado del CSV
        with open(self.log_file_path, 'w') as f:
            f.write(f"#{int(self.start_time)}\n")
            f.write("r,l,t\n")
    
    def log_episode(self, reward, length):
        """Registra la recompensa, longitud y tiempo de un episodio."""
        current_time = time.time() - self.start_time
        with open(self.log_file_path, 'a') as f:
            f.write(f"{reward},{length},{current_time}\n")


def run_custom_dqn_experiment(env_id, env_config, h_params, run_idx, wrapper_class=None):
    """
    Ejecuta una corrida de entrenamiento con el agente DQN personalizado.
    Guarda el modelo y el monitor.csv.
    """
    log_dir = f"IS-dqn_logs/{env_id}/run_{run_idx}"
    os.makedirs(log_dir, exist_ok=True)
    model_path = os.path.join(log_dir, "IS-dqn_model.pth") # <-- Guardar como .pth

    # 1. Crear el entorno
    train_params = env_config["init_params"].copy()
    train_params["render_mode"] = "ansi"
    
    env = env_config["class"](**train_params)
    if wrapper_class is not None:
        print(f"    Aplicando wrapper: {wrapper_class.__name__}")
        env = wrapper_class(env)
    
    # 2. Obtener dimensiones y dispositivo
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    Modelo creado. Dispositivo en uso: {device}")
    
    # 3. Instanciar el Agente
    agent = DQNAgent(
        state_space=state_dim,
        action_space=action_dim,
        h_params=h_params, # Pasar todos los hiperparámetros
        device=device
    )

    # 4. Inicializar Logger y Exploración
    logger = ExperimentLogger(log_dir)
    
    epsilon_start = h_params['exploration_initial_eps']
    epsilon_final = h_params['exploration_final_eps']
    epsilon_decay_steps = h_params['total_timesteps'] * h_params['exploration_fraction']
    
    def get_epsilon(step):
        if step > epsilon_decay_steps:
            return epsilon_final
        return epsilon_start - (epsilon_start - epsilon_final) * (step / epsilon_decay_steps)

    # 5. Bucle de Entrenamiento
    print(f"    Iniciando entrenamiento para {env_id} - corrida {run_idx}...")
    
    obs, info = env.reset()
    episode_reward = 0
    episode_length = 0
    recent_rewards = deque(maxlen=10) # Para imprimir progreso
    
    try:
        for step in range(1, h_params['total_timesteps'] + 1):
            
            # 1. Seleccionar acción
            epsilon = get_epsilon(step)
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            action = agent.q_net.sample_action(obs_tensor, epsilon)

            # 2. Ejecutar en el entorno
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            # 3. Almacenar transición
            done_bool = terminated or truncated
            agent.store_transition(obs, action, reward, next_obs, done_bool)

            obs = next_obs
            episode_reward += reward
            episode_length += 1

            # 4. Manejar fin de episodio
            if done_bool:
                logger.log_episode(episode_reward, episode_length)
                recent_rewards.append(episode_reward)
                
                obs, info = env.reset()
                episode_reward = 0
                episode_length = 0

            # 5. Entrenar al agente
            if step > h_params['learning_starts'] and step % h_params['train_freq'][0] == 0:
                loss = agent.update()
                
                # 6. Aplicar Soft Update (si tau está definido)
                if h_params.get('tau'):
                    agent.soft_update() # Usa el 'tau' guardado en el agente
            
            # 7. Imprimir progreso (equivalente a SimpleProgressCallback)
            if step % 5000 == 0 and len(recent_rewards) > 0:
                mean_reward = np.mean(recent_rewards)
                print(f"    [Paso {step}] Recompensa media (últimos 10 ep): {mean_reward:.2f}")

        # 8. Fin del entrenamiento
        print(f"    Entrenamiento completado para corrida {run_idx}.")
        print(f"    Guardando modelo en: {model_path}")
        agent.save_model(model_path)
        env.close()
        return model_path

    except Exception as e:
        print(f"    ERROR durante el entrenamiento para {env_id} corrida {run_idx}: {e}")
        import traceback
        traceback.print_exc() # Imprime el error completo
        env.close()
        return None


def show_custom_dqn_policy(env_id, env_config, model_path, wrapper_class=None, n_episodes=3):
    """
    Carga un modelo .pth personalizado y lo ejecuta para visualización.
    """
    print(f"\nCargando modelo DQN personalizado desde {model_path}...")
    if not os.path.exists(model_path):
        print(f"Error: No se encontró el archivo del modelo en {model_path}")
        return

    try:
        # 1. Crear entorno de visualización
        vis_params = env_config["init_params"].copy()
        vis_params["render_mode"] = "human"
        
        env = env_config["class"](**vis_params)
        if wrapper_class is not None:
            print(f"    Aplicando wrapper: {wrapper_class.__name__}")
            env = wrapper_class(env)
        
        # 2. Instanciar la red y cargar pesos
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Usamos la arquitectura guardada (asumiendo [64,64] o [128,128])
        # NOTA: Esto asume que la arquitectura está implícita o se usa la default
        # Para ser robusto, deberíamos pasar h_params['policy_kwargs']
        net_arch = [64, 64] # <-- Asumimos la default. ¡MEJORA! Pasar como arg.
        
        model = QNet(state_dim, action_dim, net_arch).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval() # Poner en modo de evaluación

        for episode in range(n_episodes):
            obs, info = env.reset()
            done = False
            total_reward = 0
            steps = 0
            print(f"\nIniciando visualización - Episodio {episode + 1}/{n_episodes}")
            
            while not done:
                # Tomar acción determinista (epsilon=0)
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
                action = model.sample_action(obs_tensor, epsilon=0.0) 
                
                obs, reward, terminated, truncated, info = env.step(action)
                
                total_reward += reward
                steps += 1
                done = terminated or truncated

            print(f"Episodio {episode + 1} terminado en {steps} pasos. Recompensa total: {total_reward:.2f}")

        env.close()
        print("\nVisualización completada.")

    except Exception as e:
        print(f"Error al cargar o ejecutar el modelo DQN para visualización: {e}")
        import traceback
        traceback.print_exc()


#########################################################################
#                                                                       #
# --- (INICIO) SECCIÓN 4: BUCLE PRINCIPAL DE EXPERIMENTACIÓN ---          #
#                                                                       #
#########################################################################

if __name__ == "__main__":
    
    check_gpu() # Verificar GPU antes de comenzar

    # --- 2. DEFINIR ENTORNOS ---
    ENVIRONMENTS = {
        "TwoTigersEnv": {
            "class": TwoTigersEnv,
            "policy": "MlpPolicy", # (Este 'policy' ya no se usa, pero lo dejamos)
            "init_params": {"max_episode_steps": 50}
        },
    }

    # --- 3. DEFINIR HIPERPARÁMETROS (para el agente custom) ---
    final_params = {
        'num_runs': 10,
        'total_timesteps': 300_000,
        
        # --- Parámetros DQN ---
        'learning_rate': 2.5e-5,     
        'buffer_size': 60_000,        
        'learning_starts': 20_000,     
        'batch_size': 2048,           
        'gamma': 0.9,                # El 'gamma' impaciente que dio el pico +49.60
        'train_freq': (4, "step"),   # (Se usa como 'update_every_steps')
        
        # --- Parámetros Soft Update ---
        'tau': 0.005,                
        'target_update_interval': 1, # (Se usa en el bucle para llamar a soft_update)
        
        # --- Parámetros de Exploración ---
        'exploration_fraction': 0.2,   
        'exploration_initial_eps': 1.0,
        'exploration_final_eps': 0.001,
        
        # --- Parámetros de Red ---
        'policy_kwargs': dict(net_arch=[128, 128]), # Red más grande
        
        # --- Parámetros PER ---
        'alpha': 0.6,
        'beta_start': 0.4,
        'beta_frames': 300_000
    }

    # --- 4. BUCLE PRINCIPAL DE ENTORNOS ---
    for env_id, env_config in ENVIRONMENTS.items():
        
        print(f"\n{'='*80}")
        print(f"🚀 Iniciando Experimento para Entorno: {env_id}")
        print(f"{'='*80}")

        # Decidir qué wrapper aplicar
        wrapper_to_apply = None
        log_prefix = "IS-dqn_logs" # Carpeta de log 
        
        if env_id == "TwoTigersEnv":
            print("!!! Aplicando BeliefStateWrapper para TwoTigersEnv !!!")
            wrapper_to_apply = BeliefStateWrapper
        
        # Rutas de logs y modelos
        base_log_dir = f"{log_prefix}/{env_id}"
        log_files_pattern = f"{base_log_dir}/run_*/monitor.csv"
        model_paths = {} # Diccionario para guardar rutas de modelos

        print(f"Iniciando experimento con Custom IS-DQN (PER) para {env_id}...")

        # --- Comprobar si los resultados ya existen ---
        valid_log_count = 0
        
        for i in range(1, final_params['num_runs'] + 1):
            f = f"{base_log_dir}/run_{i}/monitor.csv"
            m = f"{base_log_dir}/run_{i}/IS-dqn_model.pth" # <-- CAMBIO a .pth
            
            if os.path.exists(f) and os.path.exists(m):
                 try:
                     # Intenta leer el CSV para ver si es válido
                     pd.read_csv(f, skiprows=1, nrows=1) 
                     valid_log_count += 1
                     model_paths[i] = m 
                 except (pd.errors.EmptyDataError, FileNotFoundError):
                     continue # Archivo corrupto o vacío

        if valid_log_count >= final_params['num_runs']:
            print(f"Se encontraron {valid_log_count} logs y modelos. Saltando entrenamiento.")
            overall_start_time = time.time()
        else:
            print(f"Se encontraron {valid_log_count} válidos (se necesitan {final_params['num_runs']}). Iniciando entrenamiento...")
            overall_start_time = time.time()

            # --- Ejecutar los N experimentos ---
            for i in range(1, final_params['num_runs'] + 1):
                run_idx = i
                specific_log_path = f"{base_log_dir}/run_{run_idx}/monitor.csv"
                specific_model_path = f"{base_log_dir}/run_{run_idx}/IS-dqn_model.pth" # <-- CAMBIO a .pth
                
                # Comprobar si este run específico ya existe
                skip_run = False
                if os.path.exists(specific_log_path) and os.path.exists(specific_model_path):
                    try:
                        pd.read_csv(specific_log_path, skiprows=1, nrows=1)
                        print(f"\nArchivos para {env_id} corrida {run_idx} ya existen. Saltando.")
                        model_paths[run_idx] = specific_model_path
                        skip_run = True
                    except pd.errors.EmptyDataError:
                        print(f"\nLog para {env_id} corrida {run_idx} existe pero está vacío. Re-ejecutando...")

                if not skip_run:
                    run_start_time = time.time()
                    print(f"\nIniciando {env_id} corrida {run_idx}/{final_params['num_runs']}...")
                    
                    # --- LLAMADA A LA NUEVA FUNCIÓN DE ENTRENAMIENTO ---
                    saved_model_path = run_custom_dqn_experiment(
                        env_id, 
                        env_config, 
                        final_params, 
                        run_idx,
                        wrapper_to_apply 
                    )
                    
                    if saved_model_path:
                        model_paths[run_idx] = saved_model_path
                    run_time = time.time() - run_start_time
                    print(f"Corrida {run_idx} completada en {run_time:.2f} segundos.")

            print(f"\nEntrenamiento de las {final_params['num_runs']} corridas completado para {env_id}.")
            total_training_time = (time.time() - overall_start_time)
            print(f"Tiempo total de entrenamiento para {env_id}: {total_training_time/60:.2f} minutos.")


        # --- Procesar y Reportar Resultados ---
        print(f"\nProcesando resultados para {env_id}...")

        if not model_paths:
            print(f"No se encontraron logs o modelos válidos para {env_id}. Saltando reporte.")
            continue # Saltar al siguiente entorno
            
        all_episode_lengths = []
        all_episode_rewards = []
        min_episodes = float('inf')
        valid_files_processed = 0

        # Cargar datos
        for run_key in sorted(model_paths.keys()):
             f = f"{base_log_dir}/run_{run_key}/monitor.csv"
             if os.path.exists(f):
                 try:
                     df = pd.read_csv(f, skiprows=1) # Saltar la primera línea de timestamp
                     if df.empty:
                         continue
                     all_episode_lengths.append(df['l'].values)
                     all_episode_rewards.append(df['r'].values)
                     valid_files_processed += 1
                     min_episodes = min(min_episodes, len(df['l']))
                 except (pd.errors.EmptyDataError, FileNotFoundError):
                     continue

        if min_episodes == float('inf') or min_episodes == 0 or valid_files_processed == 0:
            print(f"Error: No se pudieron cargar datos válidos para {env_id}.")
            continue
        
        print(f"Procesando {valid_files_processed} logs válidos. Truncando a {min_episodes} episodios.")

        # Truncar y promediar
        run_key_to_list_index = {key: idx for idx, key in enumerate(sorted(model_paths.keys()))}
        
        padded_lengths = []
        padded_rewards = []
        for k in sorted(model_paths.keys()):
            if k in run_key_to_list_index:
                idx = run_key_to_list_index[k]
                padded_lengths.append(all_episode_lengths[idx][:min_episodes])
                padded_rewards.append(all_episode_rewards[idx][:min_episodes])

        if not padded_lengths:
             print("Error: No quedaron datos válidos después de truncar.")
             continue

        lengths_matrix = np.array(padded_lengths)
        rewards_matrix = np.array(padded_rewards)
        avg_learning_curve = np.mean(lengths_matrix, axis=0)
        avg_reward_curve = np.mean(rewards_matrix, axis=0)

        # Generar Tablas
        print("\n" + "="*65)
        print(f" " * 10 + f"REPORTE PROMEDIO DE EPISODIOS (Custom DQN-PER - {env_id})")
        print("="*65)
        print(f"{'Episodios':<15}{'Largo Promedio':<25}{'Recompensa Promedio':<25}")
        print("-"*65)
        interval = max(10, min_episodes // 10) 
        for i in range(0, len(avg_learning_curve), interval):
             if i + interval > len(avg_learning_curve): break
             episode_range = f"{i + 1}-{i + interval}"
             avg_len_interval = np.mean(avg_learning_curve[i:i + interval])
             avg_rew_interval = np.mean(avg_reward_curve[i:i + interval])
             print(f"{episode_range:<15}{avg_len_interval:<25.2f}{avg_rew_interval:<25.2f}")
        print("="*65)

        # Generar Gráficos
        print(f"\nGenerando gráficos para {env_id}...")
        window = max(1, min_episodes // 20) 
        avg_lengths_smooth = pd.Series(avg_learning_curve).rolling(window, min_periods=1).mean()
        avg_rewards_smooth = pd.Series(avg_reward_curve).rolling(window, min_periods=1).mean()
        
        # Gráfico de Largo
        plt.figure(figsize=(12, 7))
        plt.plot(avg_learning_curve, label='Promedio por episodio', alpha=0.3)
        plt.plot(avg_lengths_smooth, label=f'Media móvil (ventana={window})', color='blue')
        plt.title(f'Curva de Aprendizaje: Largo de Episodios ({env_id} - Custom IS-DQN-PER)')
        plt.xlabel('Episodios')
        plt.ylabel('Largo Promedio de Episodio')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{env_id}_IS-dqn-custom_grafico_largo_episodios.png")
        print(f"Gráfico '{env_id}_IS-dqn-custom_grafico_largo_episodios.png' guardado.")
        
        # Gráfico de Recompensa
        plt.figure(figsize=(12, 7))
        plt.plot(avg_reward_curve, label='Promedio por episodio', alpha=0.3)
        plt.plot(avg_rewards_smooth, label=f'Media móvil (ventana={window})', color='green')
        plt.title(f'Curva de Aprendizaje: Recompensa de Episodios ({env_id} - Custom IS-DQN-PER)')
        plt.xlabel('Episodios')
        plt.ylabel('Recompensa Promedio de Episodio')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{env_id}_IS-dqn-custom_grafico_recompensa_episodios.png")
        print(f"Gráfico '{env_id}_IS-dqn-custom_grafico_recompensa_episodios.png' guardado.")

        print(f"\nGráficos para {env_id} generados.")
        
        # --- Visualización ---
        # (Nota: show_custom_dqn_policy es una implementación simple,
        # necesita la arquitectura de red correcta)
        
        # if 1 in model_paths: # Visualizar el primer run
        #     print(f"\n--- Iniciando Verificación Visual del Run 1 para {env_id} ---")
        #     show_custom_dqn_policy(env_id, env_config, model_paths[1], wrapper_to_apply, n_episodes=3)
        # elif model_paths: # O el primero que exista
        #     first_run_idx = sorted(model_paths.keys())[0]
        #     print(f"\n--- Iniciando Verificación Visual del Run {first_run_idx} para {env_id} ---")
        #     show_custom_dqn_policy(env_id, env_config, model_paths[first_run_idx], wrapper_to_apply, n_episodes=3)
        # else:
        #     print(f"\nNo se encontraron modelos válidos para la verificación visual.")

    print(f"\n{'='*80}")
    print("🎉 Todos los experimentos han finalizado.")
    print("Mostrando todos los gráficos generados...")
    plt.show()