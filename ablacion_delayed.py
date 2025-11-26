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
# NOTA: ASUME QUE StochasticGridEnv y DelayedStochasticObsEnv están definidos
try:
    from Entornos.StochasticDelayedObsEnv import DelayedStochasticObsEnv 
    # Asegúrate de que la clase StochasticGridEnv también sea accesible por el wrapper
except ImportError:
    print("Error Crítico: Asegúrate de que 'Entornos/StochasticDelayedObsEnv.py' esté definido.")
    sys.exit(1)

# ==============================================================================
# 1. WRAPPERS (PSR + ENTROPÍA)
# ==============================================================================

class DelayedBeliefWrapper(gym.Wrapper):
    """ PSR Completo: Implementa Propagación + Belief + Entropía. """
    def __init__(self, env):
        super().__init__(env)
        
        self.base_env = env.base_env 
        self.H, self.W = self.base_env.size, self.base_env.size
        self.slip_prob = self.base_env.slip_prob
        self.delay_steps = self.env.delay_steps
        
        self.num_key_states = 2
        self.num_states = self.H * self.W * self.num_key_states # 200
        
        # Pre-calcular T (asumiendo que los métodos de abajo son correctos)
        self.T_matrix = self._build_transition_matrix() 
        
        # Espacio: Belief (200) + Entropía (1)
        self.observation_space = gym.spaces.Box(low=0.0, high=20.0, shape=(self.num_states + 1,), dtype=np.float32)
        self.current_belief = np.zeros(self.num_states, dtype=np.float32)

    # --- Utilitarios de Índices ---
    def _state_to_idx(self, x, y, k):
        H, W = int(self.H), int(self.W)
        if 0 <= x < H and 0 <= y < W: return int(k * (H * W) + x * W + y)
        return -1
    def _idx_to_state(self, idx):
        area = self.H * self.W; k = idx // area; rem = idx % area
        x = rem // self.W; y = rem % self.W; return x, y, k

    # --- Lógica de T (Asumiendo que el cuerpo es correcto y maneja colisiones) ---
    def _get_transition_prob(self, s, a, s_prime):
        current_x, current_y, current_k = s; next_x, next_y, next_k = s_prime
        _delta = [(-1, 0), (1, 0), (0, -1), (0, 1)]; delta = _delta[a]; T_prob = 0.0
        
        success_x_t, success_y_t = current_x + delta[0], current_y + delta[1]
        is_out_of_bounds = not (0 <= success_x_t < self.H and 0 <= success_y_t < self.W)
        if not is_out_of_bounds:
            is_wall = self.base_env._grid[success_x_t, success_y_t] == 1
            is_locked_door = (success_x_t, success_y_t) == self.base_env._door_location and not current_k
        else: is_wall, is_locked_door = False, False
        
        if is_out_of_bounds or is_wall or is_locked_door: success_x, success_y = current_x, current_y
        else: success_x, success_y = success_x_t, success_y_t
            
        failure_x, failure_y = current_x, current_y # El fallo es quedarse quieto
        
        if (next_x, next_y) == (success_x, success_y): T_prob += (1.0 - self.slip_prob)
        if (next_x, next_y) == (failure_x, failure_y): T_prob += self.slip_prob
        
        next_k_target = current_k
        if next_k == 1 and current_k == 0 and (next_x, next_y) == self.base_env._key_location: next_k_target = 1
        if next_k_target != next_k: return 0.0
            
        return T_prob

    def _build_transition_matrix(self):
        T = np.zeros((self.num_states, 4, self.num_states), dtype=np.float32)
        for s_idx in range(self.num_states):
            s = self._idx_to_state(s_idx)
            if self.base_env._grid[s[0], s[1]] == 1: continue 
            for a_idx in range(4):
                for s_prime_idx in range(self.num_states):
                    s_prime = self._idx_to_state(s_prime_idx)
                    T[s_idx, a_idx, s_prime_idx] = self._get_transition_prob(s, a_idx, s_prime)
        return T
    
    # --- Lógica de Propagación y Entropía ---
    def _rollout_belief(self, s_past_idx, action_history):
        belief = np.zeros(self.num_states, dtype=np.float32); belief[s_past_idx] = 1.0
        for a in action_history: belief = belief @ self.T_matrix[:, a, :] 
        return belief

    def _calculate_entropy(self, belief):
        b = np.clip(belief, 1e-9, 1.0); max_entropy = np.log2(self.num_states)
        return np.float32(-np.sum(b * np.log2(b)) / max_entropy)

    def reset(self, **kwargs):
        obs_past_state, info = self.env.reset(**kwargs); x, y, k = obs_past_state
        s_past_idx = self._state_to_idx(x, y, k); self.current_belief = np.zeros(self.num_states, dtype=np.float32)
        self.current_belief[s_past_idx] = 1.0; h = self._calculate_entropy(self.current_belief)
        return np.concatenate([self.current_belief, [h]]), info

    def step(self, action):
        obs_past_state, reward, terminated, truncated, info = self.env.step(action)
        x_past, y_past, k_past = obs_past_state; s_past_idx = self._state_to_idx(x_past, y_past, k_past)
        self.current_belief = self._rollout_belief(s_past_idx, self.env.action_history)
        h = self._calculate_entropy(self.current_belief)
        return np.concatenate([self.current_belief, [h]]), reward, terminated, truncated, info

class DelayedBeliefWrapperNoH(DelayedBeliefWrapper):
    """ Ablación: Solo devuelve el Belief State (200 dimensiones). """
    def __init__(self, env):
        super().__init__(env) 
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(self.num_states,), dtype=np.float32)

    def reset(self, **kwargs):
        obs_past_state, info = self.env.reset(**kwargs); x, y, k = obs_past_state
        s_past_idx = self._state_to_idx(x, y, k); self.current_belief = np.zeros(self.num_states, dtype=np.float32)
        self.current_belief[s_past_idx] = 1.0
        return self.current_belief, info

    def step(self, action):
        obs_past_state, reward, terminated, truncated, info = self.env.step(action)
        x_past, y_past, k_past = obs_past_state; s_past_idx = self._state_to_idx(x_past, y_past, k_past)
        self.current_belief = self._rollout_belief(s_past_idx, self.env.action_history)
        return self.current_belief, reward, terminated, truncated, info


# ==============================================================================
# 2. COMPONENTES DQN (AGENTE Y BUFFER PER)
# ==============================================================================

class QNet(nn.Module):
    def __init__(self, state_space: int, action_space: int, net_arch: list):
        super(QNet, self).__init__()
        self.state_space, self.action_space = state_space, action_space
        layers = []; input_dim = state_space
        for units in net_arch:
            layers.append(nn.Linear(input_dim, units)); layers.append(nn.ReLU())
            input_dim = units
        layers.append(nn.Linear(input_dim, action_space)); self.network = nn.Sequential(*layers)
    def forward(self, x): return self.network(x)
    def sample_action(self, obs, epsilon):
        if random.random() < epsilon: return random.randint(0, self.action_space - 1)
        else:
            with torch.no_grad(): return self.forward(obs).argmax().item()

class PrioritizedReplayBuffer:
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
    def sample(self, beta=0.4):
        if self.size == 0: return None, None, None
        prios = self.priorities[:self.size]; probs = prios ** self.alpha; probs /= probs.sum()
        idxs = np.random.choice(self.size, self.batch_size, p=probs, replace=True)
        samples = dict(obs=self.obs_buf[idxs], next_obs=self.next_obs_buf[idxs], acts=self.acts_buf[idxs], rews=self.rews_buf[idxs], done=self.done_buf[idxs])
        weights = (self.size * probs[idxs]) ** (-beta); weights /= weights.max(); return samples, idxs, np.array(weights, dtype=np.float32)
    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities): self.priorities[idx] = (prio + self.epsilon)
    def __len__(self): return self.size

class DQNAgent:
    def __init__(self, state_dim, action_dim, h_params, device):
        self.device, self.gamma, self.batch_size, self.tau = device, h_params['gamma'], h_params['batch_size'], h_params['tau']
        net_arch = h_params.get('policy_kwargs', {}).get('net_arch', [128, 128])
        self.q_net = QNet(state_dim, action_dim, net_arch).to(device); self.target_q_net = QNet(state_dim, action_dim, net_arch).to(device)
        self.target_q_net.load_state_dict(self.q_net.state_dict()); self.optimizer = optim.Adam(self.q_net.parameters(), lr=h_params['learning_rate'])
        self.replay_buffer = PrioritizedReplayBuffer(state_dim, h_params['buffer_size'], self.batch_size, h_params['alpha'])
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
    def save(self, path): torch.save(self.q_net.state_dict(), path)

# ==============================================================================
# 3. ARNÉS DE EXPERIMENTACIÓN (Ablación)
# ==============================================================================

class ExperimentLogger:
    def __init__(self, log_dir):
        self.path = os.path.join(log_dir, "monitor.csv"); self.start = time.time()
        with open(self.path, 'w') as f: f.write(f"#{int(self.start)}\nsteps,r,l,t\n")
    def log(self, steps, r, l):
        with open(self.path, 'a') as f: f.write(f"{steps},{r},{l},{time.time() - self.start}\n")

def check_gpu():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")
    return device

def get_experiment_config(name):
    # Aquí es donde debemos mapear los nombres a las clases y parámetros
    if name == 'Delayed-IS-DQN-PER-H': return DelayedBeliefWrapper, 0.6, 'Delayed-IS-DQN-PER-H', {'net_arch': [128, 128]}
    if name == 'Delayed-IS-DQN-H': return DelayedBeliefWrapper, 0.0, 'Delayed-IS-DQN-H', {'net_arch': [128, 128]}
    if name == 'Delayed-IS-DQN-PER': return DelayedBeliefWrapperNoH, 0.6, 'Delayed-IS-DQN-PER', {'net_arch': [128, 128]}
    if name == 'Delayed-IS-DQN': return DelayedBeliefWrapperNoH, 0.0, 'Delayed-IS-DQN', {'net_arch': [128, 128]}
    return None, None, None, None

def run_ablation_experiment(env_config, h_params, exp_name, run_idx, device):
    WrapperClass, alpha, exp_folder, net_arch_dict = get_experiment_config(exp_name)
    
    # 1. Configurar paths (guardar en carpeta con nombre del script)
    script_name = "ablacion_delayed"
    log_dir = f"{script_name}/{exp_folder}/run_{run_idx}"
    os.makedirs(log_dir, exist_ok=True)
    model_path = os.path.join(log_dir, "model.pth")
    
    # 2. Configurar hiperparámetros de la corrida
    params = h_params.copy()
    params['alpha'] = alpha
    
    # --- CORRECCIÓN DE ERROR AQUÍ (Garantizar que policy_kwargs es un dict) ---
    if params.get('policy_kwargs') is None:
        params['policy_kwargs'] = {}
        
    params['policy_kwargs'].update(net_arch_dict)

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

        agent.save(model_path)
        env.close()
        return True

    except Exception as e:
        print(f"    ERROR: {e}")
        return False


if __name__ == "__main__":
    
    check_gpu()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- CONFIGURACIÓN BASE (Ajustada para Delayed Obs Env) ---
    BASE_PARAMS = {
        'num_runs': 5,
        'total_timesteps': 500_000, 
        'learning_rate': 2.5e-5,     
        'buffer_size': 60_000,       
        'learning_starts': 20_000,   
        'batch_size': 2048,          
        'gamma': 0.95,               
        'train_freq': (4, "step"),
        'tau': 0.005,
        'target_update_interval': 1, 
        'exploration_initial_eps': 1.0,
        'exploration_final_eps': 0.001,
        'exploration_fraction': 0.3,
        'alpha': 0.6, 'beta_start': 0.4, 'beta_frames': 500_000,
        'policy_kwargs': dict(net_arch=[128, 128]), # Red base
        
        # Parámetros del Entorno
        'env_size': 10,
        'slip_prob': 0.1,
        'delay_steps': 3, 
        'max_episode_steps': 200,
    }

    ENV_CONFIG = {
        "class": DelayedStochasticObsEnv,
        "init_params": {
            "size": BASE_PARAMS['env_size'],
            "slip_prob": BASE_PARAMS['slip_prob'],
            "delay_steps": BASE_PARAMS['delay_steps'],
            "max_episode_steps": BASE_PARAMS['max_episode_steps'],
        }
    }
    
    ABLATION_CASES = [
        'Delayed-IS-DQN-PER-H',  # Completo: PSR + Entropía + PER
        'Delayed-IS-DQN-H',      # Sin PER: PSR + Entropía
        'Delayed-IS-DQN-PER',    # Sin Entropía: PSR + PER
        'Delayed-IS-DQN',        # Básico: Solo PSR
    ]

    print(f"\n{'='*80}\n🚀 INICIANDO ESTUDIO DE ABLACIÓN EN DELAYED LABERINTO (k={BASE_PARAMS['delay_steps']})\n{'='*80}")

    # Bucle de Ablación
    for case_name in ABLATION_CASES:
        for run_idx in range(1, BASE_PARAMS['num_runs'] + 1):
            
            # Comprobar si ya existe el log
            log_path = f"ablacion_delayed/{case_name}/run_{run_idx}/model.pth"
            if os.path.exists(log_path):
                print(f"\n[SKIP] {case_name} Run {run_idx} ya existe. Saltando.")
                continue

            print(f"\n--- Ejecutando {case_name} | Corrida {run_idx}/{BASE_PARAMS['num_runs']} ---")
            success = run_ablation_experiment(ENV_CONFIG, BASE_PARAMS, case_name, run_idx, device)
            
            if not success:
                print(f"La corrida {case_name} falló. Revise el traceback.")

    print("\nEstudio de Ablación Finalizado.")