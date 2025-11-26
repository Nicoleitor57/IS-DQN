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

# --- IMPORTACIÓN DE ENTORNOS ---
try:
    from Entornos.KeyDoorMazeEnv import KeyDoorMazeEnv
except ImportError:
    print("ERROR CRÍTICO: No se encuentra 'Entornos/KeyDoorMazeEnv.py'. Asegúrate de la ruta.")
    sys.exit(1)

# ==============================================================================
# 1. WRAPPERS (PSR + ENTROPÍA)
# ==============================================================================
class KeyDoorBeliefWrapper(gym.Wrapper):
    """ PSR Completo: Belief + Entropía. (Para IS-DQN-PER-H y IS-DQN-H) """
   
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


class BeliefWrapperNoH(KeyDoorBeliefWrapper):
    """ Ablación: Solo devuelve el Belief State (855 dimensiones). """
    def __init__(self, env):
        # Hereda T, Z, y lógica de update del padre
        super().__init__(env) 
        # Sobreescribir el espacio de observación a solo 855 dimensiones
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(self.num_states,), dtype=np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        sx, sy = self.base_env._start_pos; start_idx = self._state_to_idx(sx, sy, 0)
        self.current_belief = np.zeros(self.num_states, dtype=np.float32); self.current_belief[start_idx] = 1.0
        return self.current_belief, info

    def step(self, action):
        real_obs, reward, term, trunc, info = self.env.step(action)
        pred = self._predict(self.current_belief, action)
        self.current_belief = self._update(pred, real_obs)
        # Devuelve solo el belief (855 dimensiones)
        return self.current_belief, reward, term, trunc, info


# ==============================================================================
# 2. COMPONENTES DQN (RED CUSTOM + BUFFER PER)
# ==============================================================================

class QNet(nn.Module):
    """ QNet con arquitectura adaptable para input grande. """
    def __init__(self, state_space: int, action_space: int, net_arch: list):
        super(QNet, self).__init__()
        
        # El tamaño del input es 856 (con H) o 855 (sin H).
        # La red debe poder manejar ambos.
        self.state_space = state_space 
        self.action_space = action_space
        layers = []
        input_dim = state_space
        
        # Arquitectura dinámica
        for layer_size in net_arch:
            layers.append(nn.Linear(input_dim, layer_size))
            layers.append(nn.ReLU())
            input_dim = layer_size
        
        layers.append(nn.Linear(input_dim, action_space))
        self.network = nn.Sequential(*layers)

    def forward(self, x): return self.network(x)

    def sample_action(self, obs, epsilon: float):
        if random.random() < epsilon: return random.randint(0, self.action_space - 1)
        else:
            with torch.no_grad(): return self.forward(obs).argmax().item()

class PrioritizedReplayBuffer:
    # (Definición simplificada y correcta del buffer PER)
    def __init__(self, obs_dim, size, batch_size=32, alpha=0.6):
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32); self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size], dtype=np.float32); self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32); self.priorities = np.zeros(size, dtype=np.float32)
        self.max_size, self.batch_size, self.ptr, self.size, self.alpha = size, batch_size, 0, 0, alpha; self.epsilon = 1e-6
    def put(self, obs, act, rew, next_obs, done):
        max_prio = np.max(self.priorities) if self.size > 0 else 1.0; self.priorities[self.ptr] = max_prio
        self.obs_buf[self.ptr] = obs; self.next_obs_buf[self.ptr] = next_obs; self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew; self.done_buf[self.ptr] = done; self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    def sample(self, beta: float = 0.4):
        if self.size == 0: return None, None, None
        prios = self.priorities[:self.size]; probs = prios ** self.alpha; probs /= probs.sum()
        idxs = np.random.choice(self.size, self.batch_size, p=probs, replace=True)
        samples = dict(obs=self.obs_buf[idxs], next_obs=self.next_obs_buf[idxs], acts=self.acts_buf[idxs], rews=self.rews_buf[idxs], done=self.done_buf[idxs])
        weights = (self.size * probs[idxs]) ** (-beta); weights /= weights.max(); return samples, idxs, np.array(weights, dtype=np.float32)
    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities): self.priorities[idx] = (prio + self.epsilon)
    def __len__(self):
        return self.size

class DQNAgent:
    def __init__(self, state_space: int, action_space: int, h_params: dict, device='cpu'):
        self.device, self.gamma, self.batch_size, self.tau = device, h_params['gamma'], h_params['batch_size'], h_params['tau']
        net_arch = h_params.get('policy_kwargs', {}).get('net_arch', [64, 64])
        self.q_net = QNet(state_space, action_space, net_arch).to(device); self.target_q_net = QNet(state_space, action_space, net_arch).to(device)
        self.target_q_net.load_state_dict(self.q_net.state_dict()); self.optimizer = optim.Adam(self.q_net.parameters(), lr=h_params['learning_rate'])
        self.replay_buffer = PrioritizedReplayBuffer(state_space, h_params['buffer_size'], self.batch_size, h_params['alpha'])
        self.beta, self.beta_inc = h_params['beta_start'], (1.0 - h_params['beta_start']) / h_params['beta_frames']
    def store_transition(self, obs, act, rew, next_obs, done): self.replay_buffer.put(obs, act, rew, next_obs, done)
    def update(self):
        if len(self.replay_buffer) < self.batch_size: return None
        self.beta = min(1.0, self.beta + self.beta_inc); samples, idxs, weights = self.replay_buffer.sample(self.beta)
        states = torch.FloatTensor(samples['obs']).to(self.device); actions = torch.LongTensor(samples['acts'].reshape(-1,1)).to(self.device)
        rewards = torch.FloatTensor(samples['rews'].reshape(-1,1)).to(self.device); next_states = torch.FloatTensor(samples['next_obs']).to(self.device)
        dones = torch.FloatTensor(samples['done'].reshape(-1,1)).to(self.device); weights_t = torch.FloatTensor(weights).reshape(-1,1).to(self.device)
        with torch.no_grad(): q_target_max = self.target_q_net(next_states).max(1)[0].unsqueeze(1).detach()
        targets = rewards + self.gamma * q_target_max * (1 - dones); q_curr = self.q_net(states).gather(1, actions)
        elementwise_loss = F.smooth_l1_loss(q_curr, targets, reduction='none'); loss = (elementwise_loss * weights_t).mean()
        self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()
        td_errors = (targets - q_curr).abs().detach().cpu().numpy().squeeze(); self.replay_buffer.update_priorities(idxs, td_errors); return loss.item()
    def soft_update(self):
        for tp, lp in zip(self.target_q_net.parameters(), self.q_net.parameters()): tp.data.copy_(self.tau*lp.data + (1.0 - self.tau)*tp.data)
    def save_model(self, path): torch.save(self.q_net.state_dict(), path)
    def load_model(self, path): self.q_net.load_state_dict(torch.load(path, map_location=self.device)); self.target_q_net.load_state_dict(self.q_net.state_dict())


# ==============================================================================
# 3. ARNÉS DE EXPERIMENTACIÓN (Ablación)
# ==============================================================================
class ExperimentLogger:
    def __init__(self, log_dir):
        self.path = os.path.join(log_dir, "monitor.csv"); self.start = time.time()
        with open(self.path, 'w') as f: f.write(f"#{int(self.start)}\nsteps,r,l,t\n")
    def log(self, steps, r, l):
        with open(self.path, 'a') as f: f.write(f"{steps},{r},{l},{time.time() - self.start}\n")


def get_experiment_config(name):
    # Asigna la arquitectura de la red en base al requerimiento de input (855 o 856)
    if name == 'IS-DQN-PER-H': return KeyDoorBeliefWrapper, 0.6, 'IS-DQN-PER-H', {'net_arch': [512, 512]}
    if name == 'IS-DQN-H': return KeyDoorBeliefWrapper, 0.0, 'IS-DQN-H', {'net_arch': [512, 512]}
    if name == 'IS-DQN-PER': return BeliefWrapperNoH, 0.6, 'IS-DQN-PER', {'net_arch': [512, 512]}
    if name == 'IS-DQN': return BeliefWrapperNoH, 0.0, 'IS-DQN', {'net_arch': [512, 512]}
    return None, None, None, None

def run_ablation_experiment(env_config, h_params, exp_name, run_idx, device):
    WrapperClass, alpha, exp_folder, net_arch_dict = get_experiment_config(exp_name)
    
    # 1. Configurar paths (guardar en carpeta con nombre del script)
    script_name = "ablacion_maze"
    log_dir = f"{script_name}/{exp_folder}/run_{run_idx}"
    os.makedirs(log_dir, exist_ok=True)
    model_path = os.path.join(log_dir, "model.pth")
    
    # 2. Configurar hiperparámetros de la corrida
    params = h_params.copy()
    params['alpha'] = alpha
    params['policy_kwargs'].update(net_arch_dict) # Usa la arquitectura adecuada

    # 3. Inicializar entorno y wrapper
    base_env = env_config["class"](**env_config["init_params"])
    env = WrapperClass(base_env)
    
    state_dim = env.observation_space.shape[0]
    
    # 4. Instanciar agente
    agent = DQNAgent(state_dim, env.action_space.n, params, device)
    logger = ExperimentLogger(log_dir)
    
    print(f"    [ALPHA: {alpha:.1f}] Estado: {state_dim}D. Iniciando...")
    
    # 5. Bucle de Entrenamiento
    obs, _ = env.reset()
    ep_rew, ep_len = 0, 0
    eps_steps = params['total_timesteps'] * params['exploration_fraction']
    
    try:
        for t in range(1, params['total_timesteps'] + 1):
            
            # Epsilon decay
            eps = params['exploration_final_eps']
            if t < eps_steps:
                prog = t / eps_steps
                eps = 1.0 - prog * (1.0 - params['exploration_final_eps'])
            
            # Action
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
            action = agent.q_net.sample_action(obs_t, eps)
            
            # Step
            next_obs, rew, term, trunc, _ = env.step(action)
            done = term or trunc
            
            # Store and Update
            agent.store_transition(obs, action, rew, next_obs, done)
            if t > params['learning_starts'] and t % params['train_freq'][0] == 0:
                agent.update()
                agent.soft_update()
            
            obs = next_obs; ep_rew += rew; ep_len += 1
            
            if done:
                logger.log(t, ep_rew, ep_len)
                obs, _ = env.reset(); ep_rew, ep_len = 0, 0
                
            if t % 50000 == 0:
                print(f"    [Paso {t}] Epsilon: {eps:.3f}")

        agent.save_model(model_path)
        env.close()
        return True

    except Exception as e:
        print(f"    ERROR: {e}")
        return False
    
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
    


if __name__ == "__main__":
    
    check_gpu()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- CONFIGURACIÓN BASE (Ajustada para KeyDoorMazeEnv) ---
    BASE_PARAMS = {
        'num_runs': 5,
        'total_timesteps': 300_000, # Aumentamos pasos para el laberinto
        'learning_rate': 2.5e-5,     
        'buffer_size': 100_000,        
        'learning_starts': 20_000,     
        'batch_size': 2048,           
        'gamma': 0.95,               # Gamma largo plazo para el laberinto
        'train_freq': (4, "step"),   
        'tau': 0.005,                
        'target_update_interval': 1, 
        'exploration_initial_eps': 1.0,
        'exploration_final_eps': 0.01,
        'exploration_fraction': 0.3,   
        'alpha': 0.6, 'beta_start': 0.4, 'beta_frames': 500_000,
        'policy_kwargs': dict(net_arch=[512, 512]) # Red grande de 856 -> 512 -> 512
    }

    ENV_CONFIG = {
        "class": KeyDoorMazeEnv,
        "init_params": {"height": 15, "width": 19, "max_episode_steps": 200}
    }
    
    ABLATION_CASES = [
        'IS-DQN-PER-H',  # Completo: PSR + Entropía + PER
        'IS-DQN-H',      # PSR + Entropía (Sin PER)
        'IS-DQN-PER',    # PSR + PER (Sin Entropía)
        'IS-DQN',        # Básico: Solo PSR
    ]

    print(f"\n{'='*80}\n🚀 INICIANDO ESTUDIO DE ABLACIÓN EN KEYDOOR LABERINTO\n{'='*80}")

    # Bucle de Ablación
    for case_name in ABLATION_CASES:
        for run_idx in range(1, BASE_PARAMS['num_runs'] + 1):
            
            # Comprobar si ya existe el log
            log_path = f"ablacion_maze/{case_name}/run_{run_idx}/model.pth"
            if os.path.exists(log_path):
                print(f"\n[SKIP] {case_name} Run {run_idx} ya existe. Saltando.")
                continue

            print(f"\n--- Ejecutando {case_name} | Corrida {run_idx}/{BASE_PARAMS['num_runs']} ---")
            success = run_ablation_experiment(ENV_CONFIG, BASE_PARAMS, case_name, run_idx, device)
            
            if not success:
                print(f"La corrida {case_name} falló. Revise el traceback.")

    print("\nEstudio de Ablación Finalizado.")