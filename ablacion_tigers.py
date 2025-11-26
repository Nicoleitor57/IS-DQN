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
# ASUME que TwoTigersEnv está definido en Entornos/TwoTigersEnv.py
try:
    from Entornos.TwoTigersEnv import TwoTigersEnv
except ImportError:
    print("Error Crítico: No se encuentra 'Entornos/TwoTigersEnv.py'. Asegúrate de que existe.")
    sys.exit(1)


#########################################################################
# 1. WRAPPERS (PSR + ENTROPÍA)
#########################################################################

class BeliefStateWrapper(gym.Wrapper):
    """ PSR Completo: Belief + Entropía. (Para IS-DQN-PER-H y IS-DQN-H) """

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
    

class BeliefWrapperNoH(BeliefStateWrapper):
    """ Ablación: Solo devuelve el Belief State (4 dimensiones). """
    def __init__(self, env):
        super().__init__(env) 
        # Sobreescribir el espacio de observación a solo 4 dimensiones
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(self.num_states,), dtype=np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.current_belief = self.b0.copy()
        return self.current_belief, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        o_idx = self._map_obs_to_index(obs)
        b_tplus1 = self._update_belief(self.current_belief, action, o_idx)
        self.current_belief = b_tplus1
        # Devuelve solo el belief (4 dimensiones)
        return self.current_belief, reward, terminated, truncated, info


# ==============================================================================
# 2. COMPONENTES DQN (AGENTE Y BUFFER PER)
# ==============================================================================


class QNet(nn.Module):
    def __init__(self, state_space: int, action_space: int, net_arch: list = [64, 64]):
        super(QNet, self).__init__()
        
        self.state_space = state_space 
        self.action_space = action_space
        
        layers = []; input_dim = state_space
        for layer_size in net_arch:
            layers.append(nn.Linear(input_dim, layer_size)); layers.append(nn.ReLU())
            input_dim = layer_size
        layers.append(nn.Linear(input_dim, action_space)); self.network = nn.Sequential(*layers)
    def forward(self, x): return self.network(x)
    def sample_action(self, obs, epsilon: float):
        if random.random() < epsilon: return random.randint(0, self.action_space - 1)
        else:
            with torch.no_grad(): return self.forward(obs).argmax().item()

class PrioritizedReplayBuffer:
    def __init__(self, obs_dim: int, size: int, batch_size: int = 32, alpha: float = 0.6):
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
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
        dones = torch.FloatTensor(samples['done'].reshape(-1,1)).to(self.device); weights_tensor = torch.FloatTensor(weights).reshape(-1, 1).to(self.device)
        with torch.no_grad(): q_target_max = self.target_q_net(next_states).max(1)[0].unsqueeze(1).detach()
        targets = rewards + self.gamma * q_target_max * (1 - dones); q_curr = self.q_net(states).gather(1, actions)
        elementwise_loss = F.smooth_l1_loss(q_curr, targets, reduction='none'); loss = (elementwise_loss * weights_tensor).mean()
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
    

def get_experiment_config(name):
    # Devuelve el wrapper y los parámetros alpha PER para cada caso de ablación.
    if name == 'IS-DQN-PER-H': return BeliefStateWrapper, 0.6, 'IS-DQN-PER-H'
    if name == 'IS-DQN-H': return BeliefStateWrapper, 0.0, 'IS-DQN-H'
    if name == 'IS-DQN-PER': return BeliefWrapperNoH, 0.6, 'IS-DQN-PER'
    if name == 'IS-DQN': return BeliefWrapperNoH, 0.0, 'IS-DQN'
    return None, None, None

def run_ablation_experiment(env_config, h_params, exp_name, run_idx, device):
    WrapperClass, alpha, exp_folder = get_experiment_config(exp_name)
    
    # 1. Configurar paths (guardar en carpeta con nombre del script)
    script_name = "ablacion_tigers"
    log_dir = f"{script_name}/{exp_folder}/run_{run_idx}"
    os.makedirs(log_dir, exist_ok=True)
    model_path = os.path.join(log_dir, "model.pth")
    
    # 2. Configurar hiperparámetros de la corrida
    params = h_params.copy()
    params['alpha'] = alpha # Alpha determina si PER está activo (alpha > 0)
    
    # 3. Inicializar entorno y wrapper
    base_env = env_config["class"](**env_config["init_params"])
    env = WrapperClass(base_env)
    
    # Ajustar la arquitectura de la red al tamaño del estado (4 o 5)
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


if __name__ == "__main__":
    
    check_gpu()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- CONFIGURACIÓN BASE (Ajustada a tu mejor corrida estable) ---
    BASE_PARAMS = {
        'num_runs': 5,
        'total_timesteps': 300_000,
        'learning_rate': 2.5e-5,     
        'buffer_size': 60_000,        
        'learning_starts': 20_000,     
        'batch_size': 2048,           
        'gamma': 0.9,              # Gamma balanceado
        'train_freq': (4, "step"),   
        'tau': 0.005,                
        'target_update_interval': 1, 
        'exploration_initial_eps': 1.0,
        'exploration_final_eps': 0.001,
        'exploration_fraction': 0.2,   
        'alpha': 0.6, 'beta_start': 0.4, 'beta_frames': 300_000,
        'policy_kwargs': dict(net_arch=[64, 64]) 
    }

    ENV_CONFIG = {
        "class": TwoTigersEnv,
        "init_params": {"max_episode_steps": 50}
    }
    
    ABLATION_CASES = [
        'IS-DQN-PER-H',  # Completo (PSR+H+PER)
        'IS-DQN-H',      # Sin PER (PSR+H)
        'IS-DQN-PER',    # Sin Entropía (PSR+PER)
        'IS-DQN',        # Básico (Solo PSR)
    ]

    print(f"\n{'='*80}\n🚀 INICIANDO ESTUDIO DE ABLACIÓN EN {BASE_PARAMS['num_runs']} CORRIDAS\n{'='*80}")

    # Bucle de Ablación
    for case_name in ABLATION_CASES:
        for run_idx in range(1, BASE_PARAMS['num_runs'] + 1):
            
            # Comprobar si ya existe el log
            log_path = f"ablacion_tigers/{case_name}/run_{run_idx}/monitor.csv"
            if os.path.exists(log_path):
                try:
                    df = pd.read_csv(log_path, skiprows=1)
                    if not df.empty:
                        print(f"\n[SKIP] {case_name} Run {run_idx} ya existe y es válido. Saltando.")
                        continue
                except:
                    pass

            print(f"\n--- Ejecutando {case_name} | Corrida {run_idx}/{BASE_PARAMS['num_runs']} ---")
            success = run_ablation_experiment(ENV_CONFIG, BASE_PARAMS, case_name, run_idx, device)
            
            if not success:
                print(f"La corrida {case_name} falló. Revise el traceback.")