import gymnasium as gym
import numpy as np
# Asumimos que tu entorno se llama TwoTigersEnv y está en TwoTigersEnv.py
from Entornos.TwoTigersEnv import TwoTigersEnv 

class BeliefStateWrapper(gym.Wrapper):
    """
    Wrapper que transforma el POMDP TwoTigersEnv en un MDP sobre el belief state.
    
    *** VERSIÓN CORREGIDA (v2) ***
    
    Asume que el entorno base (env):
    1. Tiene un espacio de acciones: gym.spaces.Discrete(9)
    2. Devuelve observaciones: un array [o1, o2] donde o1, o2 in {0, 1, 2}.
       - 0: OL (Oído Izquierda)
       - 1: OR (Oído Derecha)
       - 2: NO_OBS (Acción no fue 'Escuchar')
    """

    def __init__(self, env):
        super().__init__(env)
        
        # --- 1. Definir constantes del POMDP ---
        self.num_states = 4  # |S| = 4  [(SL,SL), (SL,SR), (SR,SL), (SR,SR)]
        self.num_actions = 9 # |A| = 9  [(AE,AE), (AE,AL), ..., (AR,AR)]
        
        # ¡CORRECCIÓN! |O_i| = 3, por lo tanto |O| = 3x3 = 9
        self.num_obs = 9     
        
        # --- 2. Construir Modelos (T y Z) ---
        self.T = self._build_transition_model() # Matriz [s, a, s'] (4x9x4)
        self.Z = self._build_observation_model()  # Matriz [s', a, o] (4x9x9)
        
        # --- 3. Definir Belief Inicial (b0) ---
        self.b0 = np.full(self.num_states, 1.0 / self.num_states, dtype=np.float32)
        self.current_belief = self.b0.copy()

        # --- 4. Sobrescribir el Espacio de Observación ---
        # El agente DQN verá el belief (vector de 4 flotantes)
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.num_states + 1,),
            dtype=np.float32
        )
        
    def _calculate_entropy(self, belief):
        """Calcula la entropía de Shannon del vector de belief."""
        # Añadimos 1e-9 para evitar log(0)
        belief = np.clip(belief, 1e-9, 1.0)
        entropy = -np.sum(belief * np.log2(belief))
        return np.float32(entropy)
    

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.current_belief = self.b0.copy()
        
        # (Opcional pero recomendado) Si el env da una obs inicial
        # que NO es [2, 2], podríamos usarla para actualizar b0.
        # Por ahora, simplemente devolvemos el 'prior' b0.
        
        current_entropy = self._calculate_entropy(self.current_belief)
        obs_with_entropy = np.concatenate([self.current_belief, [current_entropy]])
        
        return obs_with_entropy, info

    def step(self, action):
        # 1. 'action' es el a_idx (0-8)
        
        # 2. Ejecuta el paso en el entorno real
        # 'obs' será un array, p.ej., [0, 1], [1, 2], [2, 2], etc.
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # 3. Mapea la observación real a su índice (0-8)
        o_idx = self._map_obs_to_index(obs)
        
        # 4. Actualiza el belief state
        b_t = self.current_belief
        a_idx = action
        b_tplus1 = self._update_belief(b_t, a_idx, o_idx)
        
        # 5. Guarda el nuevo belief
        self.current_belief = b_tplus1
        
        current_entropy = self._calculate_entropy(self.current_belief)
        obs_with_entropy = np.concatenate([self.current_belief, [current_entropy]])
        
        # 6. Devuelve el *nuevo belief* como el "estado"
        return obs_with_entropy, reward, terminated, truncated, info

    # --- Motor del Filtro Bayesiano ---

    def _update_belief(self, b_t, a_idx, o_idx):
        """
        Actualiza el belief state usando el filtro Bayesiano.
        b_{t+1}(s') = η * Z(o | s', a) * Σ_{s} [ T(s' | s, a) * b_t(s) ]
        """
        
        # --- 1. Predicción ---
        T_a = self.T[:, a_idx, :]  # Matriz T(s, s' | a) de forma (4, 4)
        b_hat = T_a.T @ b_t        # Vector de forma (4,)
        
        # --- 2. Actualización ---
        # self.Z tiene forma [s', a, o] -> (4, 9, 9)
        Z_o = self.Z[:, a_idx, o_idx] # Vector de forma (4,)
        
        b_new_unnormalized = Z_o * b_hat
        
        # --- 3. Normalización ---
        prob_obs = np.sum(b_new_unnormalized)
        
        if prob_obs < 1e-9:
            # Observación "imposible"
            return self.b0.copy()
        else:
            b_tplus1 = b_new_unnormalized / prob_obs
            return b_tplus1

    def _map_obs_to_index(self, obs):
        """
        Función de ayuda para convertir la observación del entorno [o1, o2]
        (donde o1, o2 in {0,1,2}) a un índice único (0-8).
        """
        
        # Caso 1: Array/lista (p.ej., [1, 2])
        if isinstance(obs, (list, np.ndarray)) and len(obs) == 2:
            o1, o2 = obs
            
            # ¡CORRECCIÓN! Usamos base-3 para el mapeo
            # [0, 0] -> 0*3 + 0 = 0
            # [0, 1] -> 0*3 + 1 = 1
            # [0, 2] -> 0*3 + 2 = 2
            # [1, 0] -> 1*3 + 0 = 3
            # ...
            # [1, 2] -> 1*3 + 2 = 5  (¡Tu error!)
            # ...
            # [2, 2] -> 2*3 + 2 = 8
            o_idx = int(o1) * 3 + int(o2)
            
            if 0 <= o_idx < self.num_obs: # self.num_obs es 9
                return o_idx
            else:
                raise ValueError(f"Observación (array) con valores inválidos: {obs}.")

        # Caso 2: El env ya devolvió el índice (raro, pero seguro)
        if isinstance(obs, (int, np.integer)):
            if 0 <= obs < self.num_obs:
                return obs
            
        raise ValueError(f"Formato de observación no reconocida: {obs}.")

    # --- Funciones de Construcción de Modelos (Helpers) ---
    def _build_observation_model(self):
            """
            ¡CORREGIDO (v3)!
            Construye el tensor Z(s', a, o) = P(o | s', a) de 4x9x9.
            Alineado con el mapeo del entorno:
            {0: NO_OBS, 1: OL, 2: OR}
            """
            
            # Mapeos (solo para construcción)
            map_s_i = {'SL': 0, 'SR': 1}
            map_a_i = {'AE': 0, 'AL': 1, 'AR': 2}
            
            # --- ¡CAMBIO 1! ---
            # Mapeo de observación corregido para que coincida con TwoTigersEnv.py
            map_o_i = {'NO_OBS': 0, 'OL': 1, 'OR': 2}
            
            map_s = {0: ('SL', 'SL'), 1: ('SL', 'SR'), 2: ('SR', 'SL'), 3: ('SR', 'SR')}
            map_a = {
                0: ('AE', 'AE'), 1: ('AE', 'AL'), 2: ('AE', 'AR'),
                3: ('AL', 'AE'), 4: ('AR', 'AE'), 5: ('AL', 'AL'),
                6: ('AL', 'AR'), 7: ('AR', 'AL'), 8: ('AR', 'AR')
            }
            
            # --- ¡CAMBIO 2! ---
            # Mapeo de observación CONJUNTA corregido (ahora 0*3+0 es NO_OBS, NO_OBS)
            map_o = {
                0: ('NO_OBS', 'NO_OBS'), 1: ('NO_OBS', 'OL'), 2: ('NO_OBS', 'OR'),
                3: ('OL', 'NO_OBS'), 4: ('OL', 'OL'), 5: ('OL', 'OR'),
                6: ('OR', 'NO_OBS'), 7: ('OR', 'OL'), 8: ('OR', 'OR')
            }

            # 1. Tensor Z_i base (2x3x3) -> Z_i[s_i, a_i, o_i]
            Z_i = np.zeros((2, 3, 3))
            
            # --- ¡CAMBIO 3! ---
            # Construcción del tensor Z_i corregida
            
            # Acción = 0 (AE - Escuchar): Devuelve OL (1) o OR (2)
            # P(o|s=SL, a=AE) -> {NO_OBS(0): 0.0, OL(1): 0.85, OR(2): 0.15}
            Z_i[0, 0, :] = [0.0, 0.85, 0.15]
            # P(o|s=SR, a=AE) -> {NO_OBS(0): 0.0, OL(1): 0.15, OR(2): 0.85}
            Z_i[1, 0, :] = [0.0, 0.15, 0.85]
            
            # Acción = 1 (AL - Abrir Izq): Devuelve NO_OBS (0)
            # P(o|s=*, a=AL) -> {NO_OBS(0): 1.0, OL(1): 0.0, OR(2): 0.0}
            Z_i[:, 1, :] = [1.0, 0.0, 0.0]
            
            # Acción = 2 (AR - Abrir Der): Devuelve NO_OBS (0)
            # P(o|s=*, a=AR) -> {NO_OBS(0): 1.0, OL(1): 0.0, OR(2): 0.0}
            Z_i[:, 2, :] = [1.0, 0.0, 0.0]

            # 2. Tensor Z final (4x9x9) -> Z[s', a, o]
            Z = np.zeros((self.num_states, self.num_actions, self.num_obs))
            
            for s_prime_idx in range(self.num_states):
                for a_idx in range(self.num_actions):
                    for o_idx in range(self.num_obs):
                        
                        s1_str, s2_str = map_s[s_prime_idx]
                        a1_str, a2_str = map_a[a_idx]
                        o1_str, o2_str = map_o[o_idx]
                        
                        s1_i = map_s_i[s1_str]
                        s2_i = map_s_i[s2_str]
                        a1_i = map_a_i[a1_str]
                        a2_i = map_a_i[a2_str]
                        o1_i = map_o_i[o1_str]
                        o2_i = map_o_i[o2_str]
                        
                        prob_z1 = Z_i[s1_i, a1_i, o1_i]
                        prob_z2 = Z_i[s2_i, a2_i, o2_i]
                        
                        Z[s_prime_idx, a_idx, o_idx] = prob_z1 * prob_z2
            
            return Z

    def _build_transition_model(self):
        """
        Construye el tensor T(s, a, s') = P(s' | s, a) de 4x9x4.
        (Esta función estaba bien y no necesita cambios).
        """
        # Mapeos
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
                    
                    s1_i = map_s_i[s1_str]
                    s2_i = map_s_i[s2_str]
                    a1_i = map_a_i[a1_str]
                    a2_i = map_a_i[a2_str]
                    s1_p_i = map_s_i[s1_p_str]
                    s2_p_i = map_s_i[s2_p_str]
                    
                    prob_t1 = T_i[s1_i, a1_i, s1_p_i]
                    prob_t2 = T_i[s2_i, a2_i, s2_p_i]
                    
                    T[s_idx, a_idx, s_prime_idx] = prob_t1 * prob_t2
        
        return T
    
    
    
import torch
from collections import deque
from IS.PER_DQN import DQNAgent

# 1. Crear el entorno
env = TwoTigersEnv(max_episode_steps=50)
env = BeliefStateWrapper(env) # ¡Tu wrapper PSR+Entropía!

# 2. Obtener tamaños de estado y acción
state_dim = env.observation_space.shape[0] # Debería ser 5
action_dim = env.action_space.n           # Debería ser 9

# 3. Usar los hiperparámetros que SÍ funcionaron
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Esta es la configuración que te dio el pico +49.60, adaptada
agent = DQNAgent(
    state_space=state_dim,
    action_space=action_dim,
    device=device,
    lr=2.5e-4,
    gamma=0.9, # El gamma "impaciente" que funcionó
    batch_size=512,
    buffer_size=60_000,
    alpha=0.6, # Valor estándar de PER
    beta_start=0.4, # Valor estándar de PER
    beta_frames=300_000 # Anneal durante todo el entrenamiento
)

# 4. Hiperparámetros de exploración
epsilon_start = 1.0
epsilon_final = 0.001 # El ruido final muy bajo
epsilon_decay_steps = 300_000 * 0.2 # 20% de los pasos

def get_epsilon(step):
    if step > epsilon_decay_steps:
        return epsilon_final
    return epsilon_start - (epsilon_start - epsilon_final) * (step / epsilon_decay_steps)

num_total_steps = 300_000
learning_starts = 20_000
update_every_steps = 4
target_update_tau = 0.005 # ¡Usemos Soft Updates!

obs, info = env.reset()
episode_reward = 0
episode_rewards = [] # Para logging

for step in range(1, num_total_steps + 1):
    # 1. Seleccionar acción
    epsilon = get_epsilon(step)
    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
    action = agent.q_net.sample_action(obs_tensor, epsilon)

    # 2. Ejecutar en el entorno
    next_obs, reward, terminated, truncated, info = env.step(action)

    # 3. Almacenar transición
    agent.store_transition(obs, action, reward, next_obs, terminated or truncated)

    obs = next_obs
    episode_reward += reward

    # 4. Manejar fin de episodio
    if terminated or truncated:
        print(f"Paso {step}, Recompensa Episodio: {episode_reward}")
        episode_rewards.append(episode_reward)
        obs, info = env.reset()
        episode_reward = 0

        # (Aquí puedes imprimir la recompensa media de los últimos 10)
        if len(episode_rewards) > 10:
            print(f"    Media(10): {np.mean(episode_rewards[-10:])}")

    # 5. Entrenar al agente (si hay suficientes datos)
    if step > learning_starts and step % update_every_steps == 0:
        loss = agent.update()

        # 6. Aplicar Soft Update (mucho más estable)
        agent.soft_update(tau=target_update_tau)

# (Guardar modelo, graficar, etc.)