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
            shape=(self.num_states,),
            dtype=np.float32
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.current_belief = self.b0.copy()
        
        # (Opcional pero recomendado) Si el env da una obs inicial
        # que NO es [2, 2], podríamos usarla para actualizar b0.
        # Por ahora, simplemente devolvemos el 'prior' b0.
        
        return self.current_belief, info

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
        
        # 6. Devuelve el *nuevo belief* como el "estado"
        return self.current_belief, reward, terminated, truncated, info

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
    
    def _calculate_entropy(self, belief):
        """Calcula la entropía de Shannon del vector de belief."""
        # Añadimos 1e-9 para evitar log(0)
        belief = np.clip(belief, 1e-9, 1.0)
        entropy = -np.sum(belief * np.log2(belief))
        return np.float32(entropy)
    
    
#test clase BeliefStateWrapper

# print("Probando BeliefStateWrapper...")
# env = TwoTigersEnv(max_episode_steps=1000)
# wrapped_env = BeliefStateWrapper(env)
# print(wrapped_env._build_observation_model())
# print(wrapped_env._build_transition_model())
# print("\n--- Probando el 'reset' y 'step' del wrapper ---")
# try:
#     print("Probando wrapped_env.reset()...")
#     obs, info = wrapped_env.reset()
#     print(f"Observación inicial (belief): {obs.shape}")

#     print("\nProbando wrapped_env.step(0)...")
#     # Tomamos una acción de ejemplo (índice 0 = 'AE', 'AE')
#     action = 0 
#     new_obs, reward, terminated, truncated, info = wrapped_env.step(action)
#     print("¡El step funcionó!")

# except ValueError as e:
#     print("\n¡FALLÓ! Se replicó el error:")
#     print(e)


# model = DQN("MlpPolicy", wrapped_env, verbose=1)
# model.learn(total_timesteps=300000)




import gymnasium as gym
import numpy as np
import pandas as pd
import time
import os
import glob
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

import torch 
from stable_baselines3.common.callbacks import BaseCallback

# --- 1. IMPORTAR ENTORNOS PERSONALIZADOS ---
# (Asegúrate de que estos archivos .py estén en la misma carpeta)
#from Entornos.PODoorEnv import POKeyDoorEnv
#from Entornos.KeyDoorMazeEnv import KeyDoorMazeEnv
from Entornos.TwoTigersEnv import TwoTigersEnv
#from Entornos.DelayedObsEnv import DelayedObsEnv

def check_gpu():
    """
    Verifica si PyTorch (backend de SB3) puede detectar y usar la GPU.
    """
    print(f"\n{'='*40}")
    print(" Verificando disponibilidad de GPU (PyTorch)...")
    print(f"{'='*40}")
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        current_dev_idx = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_dev_idx)
        
        print(f"✅ ¡Éxito! GPU disponible.")
        print(f"   Dispositivos encontrados: {device_count}")
        print(f"   Usando dispositivo {current_dev_idx}: {device_name}")
    else:
        print(f"❌ GPU no disponible.")
        print("   El entrenamiento se ejecutará en la CPU (será más lento).")
    print(f"{'='*40}\n")
    
class SimpleProgressCallback(BaseCallback):
    """
    Un callback simple que imprime la recompensa media de los últimos 10 episodios.
    """
    def __init__(self, check_freq: int, verbose: int = 1):
        super(SimpleProgressCallback, self).__init__(verbose)
        self.check_freq = check_freq

    def _on_step(self) -> bool:
        # Revisa si es el momento de imprimir (cada `check_freq` pasos)
        if self.num_timesteps % self.check_freq == 0:
            
            # 'self.training_env' es el DummyVecEnv
            # 'self.training_env.envs[0]' es el wrapper Monitor
            monitor = self.training_env.envs[0]
            
            # Obtiene las recompensas de los últimos 10 episodios (si existen)
            recent_rewards = monitor.get_episode_rewards()[-10:]
            
            if recent_rewards:
                mean_reward = np.mean(recent_rewards)
                # Imprime el progreso
                print(f"    [Paso {self.num_timesteps}] Recompensa media (últimos 10 ep): {mean_reward:.2f}")
        
        return True # Importante: devuelve True para continuar el entrenamiento


def run_dqn_experiment(env_id, env_config, h_params, run_idx, wrapper_class=None): ### <-- CAMBIO: Añadido 'wrapper_class'
    """
    Ejecuta una corrida de entrenamiento de DQN para un entorno dado.
    MODIFICADO: Acepta config. del entorno y guarda logs/modelos en carpetas específicas.

    :param env_id: String con el nombre del entorno (para carpetas).
    :param env_config: Diccionario con "class", "policy", "init_params".
    :param h_params: Diccionario con hiperparámetros de DQN.
    :param run_idx: El índice de la corrida actual.
    :param wrapper_class: (Opcional) La clase del wrapper a aplicar (p.ej. BeliefStateWrapper) ### <-- CAMBIO
    :returns: Ruta al modelo guardado o None si falla.
    """
    log_dir = f"IS-dqn_logs/{env_id}/run_{run_idx}"
    os.makedirs(log_dir, exist_ok=True)
    model_path = os.path.join(log_dir, "IS-dqn_model.zip")

    train_params = env_config["init_params"].copy()
    train_params["render_mode"] = "ansi"
    
    def make_env():
        env = env_config["class"](**train_params)
        
        ### --- (INICIO) CAMBIO: APLICAR WRAPPER AQUÍ --- ###
        if wrapper_class is not None:
            print(f"    Aplicando wrapper: {wrapper_class.__name__}")
            env = wrapper_class(env)
        ### --- (FIN) CAMBIO --- ###
            
        env = Monitor(env, filename=log_dir)
        return env

    env = DummyVecEnv([make_env])

    # Determinar la política. Si usamos el wrapper, forzamos MlpPolicy
    # porque el wrapper convierte la obs a un vector Box (flotantes)
    policy_to_use = env_config["policy"]
    if wrapper_class is not None:
        policy_to_use = "MlpPolicy"
        print(f"    Wrapper detectado. Forzando uso de 'MlpPolicy'.")
        
    # Definir el modelo DQN
    model = DQN(
        policy_to_use, ### <-- CAMBIO: Usar 'policy_to_use'
        env,
        learning_rate=h_params['learning_rate'],
        buffer_size=h_params['buffer_size'],
        learning_starts=h_params['learning_starts'],
        batch_size=h_params['batch_size'],
        gamma=h_params['gamma'],
        train_freq=h_params['train_freq'],
        target_update_interval=h_params['target_update_interval'],
        exploration_fraction=h_params['exploration_fraction'],
        exploration_final_eps=h_params['exploration_final_eps'],
        policy_kwargs=h_params['policy_kwargs'],
        verbose=0,
        device="auto" # 'auto' le dice a SB3 que use GPU si está disponible
    )
    
    # --- VERIFICACIÓN DE DISPOSITIVO (GPU/CPU) ---
    print(f"    Modelo creado. Dispositivo en uso: {model.device}")
    # ----------------------------------------------
    
    # --- DEFINIR EL CALLBACK DE PROGRESO ---
    # Imprimirá el progreso cada 5000 pasos
    progress_callback = SimpleProgressCallback(check_freq=5000)
    # ---------------------------------------

    print(f"    Iniciando entrenamiento para {env_id} - corrida {run_idx}...")
    try:
        # --- PASAR EL CALLBACK A .learn() ---
        model.learn(
            total_timesteps=h_params['total_timesteps'],
            callback=progress_callback 
        )
        # --------------------------------------
        
        print(f"    Entrenamiento completado para corrida {run_idx}.")
        print(f"    Guardando modelo en: {model_path}")
        model.save(model_path)
        env.close()
        return model_path
    except Exception as e:
        print(f"    ERROR durante el entrenamiento para {env_id} corrida {run_idx}: {e}")
        env.close()
        return None


def show_dqn_policy(env_id, env_config, model_path, wrapper_class=None, n_episodes=3): ### <-- CAMBIO
    """
    Carga un modelo DQN guardado y lo ejecuta en el entorno para visualización.
    MODIFICADO: Acepta config. del entorno para instanciación.
    """
    print(f"\nCargando modelo DQN desde {model_path} para visualización...")
    if not os.path.exists(model_path):
        print(f"Error: No se encontró el archivo del modelo en {model_path}")
        return

    try:
        model = DQN.load(model_path)
        
        # --- Instanciar entorno personalizado para visualización ---
        vis_params = env_config["init_params"].copy()
        vis_params["render_mode"] = "human" # Forzar modo 'human'
        
        def make_vis_env(): ### <-- CAMBIO: Usar una función para aplicar wrapper
            env = env_config["class"](**vis_params)
            if wrapper_class is not None:
                print(f"    Aplicando wrapper: {wrapper_class.__name__}")
                env = wrapper_class(env)
            return env
        
        # Envolver en DummyVecEnv para que coincida con la forma de entrenamiento
        env = DummyVecEnv([make_vis_env]) ### <-- CAMBIO

        for episode in range(n_episodes):
            obs = env.reset() # reset() en VecEnv devuelve la obs
            done = False
            total_reward = 0
            steps = 0
            print(f"\nIniciando visualización - Episodio {episode + 1}/{n_episodes}")
            
            # El bucle de un VecEnv es ligeramente diferente
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, info = env.step(action)
                
                # En VecEnv, 'terminated' es un array, e 'info' también
                total_reward += reward[0]
                steps += 1
                
                # 'done' es True si el primer (y único) env terminó
                done = terminated[0] 
                
                # 'truncated' puede estar dentro de 'info'
                if info[0].get('TimeLimit.truncated', False):
                    done = True

            print(f"Episodio {episode + 1} terminado en {steps} pasos. Recompensa total: {total_reward:.2f}")

        env.close()
        print("\nVisualización completada.")

    except Exception as e:
        print(f"Error al cargar o ejecutar el modelo DQN para visualización: {e}")


if __name__ == "__main__":
    
    check_gpu() # Verificar GPU antes de comenzar

    # --- 2. DEFINIR LOS ENTORNOS A PROBAR ---
    # Mapea un ID a la clase del entorno, su política y sus parámetros de __init__
    ENVIRONMENTS = {
        # "POKeyDoorEnv": {
        #     "class": POKeyDoorEnv,
        #     "policy": "MlpPolicy", # 3x3 grid será aplanado
        #     "init_params": {"size": 10, "max_episode_steps": 1000}
        # },
        # "KeyDoorMazeEnv": {
        #     "class": KeyDoorMazeEnv,
        #     "policy": "MlpPolicy", # 3x3 grid será aplanado
        #     "init_params": {"height": 15, "width": 19, "max_episode_steps": 1000}
        # },
        "TwoTigersEnv": {
            "class": TwoTigersEnv,
            "policy": "MlpPolicy", # La política se cambiará a MlpPolicy si se usa el wrapper
            "init_params": {"max_episode_steps": 50}
        },
        # "DelayedObsEnv": {
        #     "class": DelayedObsEnv,
        #     "policy": "MlpPolicy", # 3x3 grid será aplanado
        #     "init_params": {"size": 10, "delay_steps": 3, "max_episode_steps": 1000}
        # }
    }

    # --- 3. DEFINIR HIPERPARÁMETROS ---
    # (Estos son genéricos, pueden necesitar ajuste por entorno)
    final_params = {
        'num_runs': 10,
        'total_timesteps': 300000,
        'learning_rate': 2.5e-4,     
        'buffer_size': 60_000,        
        'learning_starts': 20_000,     
        'batch_size': 128,           
        'gamma': 0.95,               
        'train_freq': (4, "step"),   
        'target_update_interval': 10_000, 
        'exploration_fraction': 0.2,   
        'exploration_final_eps': 0.01, 
        'policy_kwargs': dict(net_arch=[64,64])
    }

    # --- 4. BUCLE PRINCIPAL DE ENTORNOS ---
    for env_id, env_config in ENVIRONMENTS.items():
        
        ### --- (INICIO) CAMBIO: Lógica para aplicar wrappers --- ###
        
        # NO CREAR EL WRAPPER AQUÍ
        # 1. NO sobrescribir env_id:
        # env_id = BeliefStateWrapper(env_config["class"](**env_config["init_params"])) # <-- ESTO ES INCORRECTO
        
        print(f"\n{'='*80}")
        print(f"🚀 Iniciando Experimento para Entorno: {env_id}") # <-- Ahora env_id es el string correcto
        print(f"{'='*80}")

        # 2. Decidir qué wrapper aplicar (si aplica)
        wrapper_to_apply = None
        log_prefix = "IS-dqn_logs" # Carpeta de log normal
        
        if env_id == "TwoTigersEnv":
            print("!!! Aplicando BeliefStateWrapper para TwoTigersEnv !!!")
            wrapper_to_apply = BeliefStateWrapper
            log_prefix = "IS-dqn_logs" # Usar una carpeta de log diferente para el experimento
        
        ### --- (FIN) CAMBIO --- ###


        # Rutas de logs y modelos específicas del entorno
        base_log_dir = f"{log_prefix}/{env_id}" ### <-- CAMBIO: log_prefix
        log_files_pattern = f"{base_log_dir}/run_*/monitor.csv"
        model_paths = {} # Resetear para cada entorno

        print(f"Iniciando experimento con IS-DQN para {env_id}...")

        # --- Comprobar si los resultados ya existen ---
        existing_log_files = glob.glob(log_files_pattern)
        valid_log_count = 0
        
        for i in range(1, final_params['num_runs'] + 1):
            f = f"{base_log_dir}/run_{i}/monitor.csv"
            m = f"{base_log_dir}/run_{i}/IS-dqn_model.zip"
            if os.path.exists(f) and os.path.exists(m):
                 try:
                     pd.read_csv(f, skiprows=1, nrows=1)
                     valid_log_count += 1
                     model_paths[i] = m 
                 except (pd.errors.EmptyDataError, FileNotFoundError):
                     continue 

        if valid_log_count >= final_params['num_runs']:
            print(f"Se encontraron {valid_log_count} archivos de log y modelos existentes para {env_id}. Saltando la fase de entrenamiento.")
            overall_start_time = time.time()
        else:
            print(f"Se encontraron {valid_log_count} archivos válidos para {env_id} (se necesitan {final_params['num_runs']}). Iniciando entrenamiento...")
            print(f"Corriendo {final_params['num_runs']} veces con {final_params['total_timesteps']} timesteps cada una.")
            overall_start_time = time.time()
            all_run_times = []

            # --- Ejecutar los 30 experimentos ---
            for i in range(final_params['num_runs']):
                run_idx = i + 1
                specific_log_path = f"{base_log_dir}/run_{run_idx}/monitor.csv"
                specific_model_path = f"{base_log_dir}/run_{run_idx}/dqn_model.zip"
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
                    
                    saved_model_path = run_dqn_experiment(
                        env_id, 
                        env_config, 
                        final_params, 
                        run_idx,
                        wrapper_to_apply ### <-- CAMBIO: Pasar el wrapper
                    )
                    
                    if saved_model_path:
                        model_paths[run_idx] = saved_model_path
                    run_time = time.time() - run_start_time
                    all_run_times.append(run_time)
                    print(f"Corrida {run_idx} completada en {run_time:.2f} segundos.")

            print(f"\nEntrenamiento de las {final_params['num_runs']} corridas completado para {env_id}.")
            total_training_time = (time.time() - overall_start_time)
            print(f"Tiempo total de entrenamiento para {env_id}: {total_training_time/60:.2f} minutos.")


        # --- Procesar y Reportar Resultados ---
        print(f"\nProcesando resultados para {env_id}...")

        log_files = glob.glob(log_files_pattern)
        if not model_paths: # Si no se encontraron modelos válidos
            print(f"No se encontraron archivos de log o modelos válidos para {env_id}. Saltando reporte.")
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
                     df = pd.read_csv(f, skiprows=1)
                     if df.empty:
                         print(f"Advertencia: El archivo {f} está vacío y será ignorado.")
                         continue
                     all_episode_lengths.append(df['l'].values)
                     all_episode_rewards.append(df['r'].values)
                     valid_files_processed += 1
                     min_episodes = min(min_episodes, len(df['l']))
                 except (pd.errors.EmptyDataError, FileNotFoundError):
                     print(f"Advertencia: No se pudo leer {f} aunque existe.")
             else:
                 print(f"Advertencia: Falta el archivo {f} para la corrida {run_key}.")


        if min_episodes == float('inf') or min_episodes == 0 or valid_files_processed == 0:
            print(f"Error: No se pudieron cargar datos válidos para {env_id}. No se pueden generar reportes.")
            continue # Saltar al siguiente entorno
        
        print(f"Procesando {valid_files_processed} logs válidos para {env_id}.")
        print(f"Truncando a {min_episodes} episodios.")

        # Truncar y promediar
        # Mapear run_key (1..30) al índice de la lista (0..N-1)
        run_key_to_list_index = {key: idx for idx, key in enumerate(sorted(model_paths.keys()))}
        
        padded_lengths = []
        padded_rewards = []
        for k in sorted(model_paths.keys()):
            # Chequear si la key existe en el mapeo (maneja logs/corridas faltantes)
            if k in run_key_to_list_index: ### <-- CAMBIO: Chequeo de seguridad
                idx = run_key_to_list_index[k]
                if len(all_episode_lengths[idx]) >= min_episodes:
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
        print(f" " * 10 + f"REPORTE PROMEDIO DE EPISODIOS (DQN - {env_id})")
        print("="*65)
        print(f"{'Episodios':<15}{'Largo Promedio':<25}{'Recompensa Promedio':<25}")
        print("-"*65)
        interval = max(10, min_episodes // 10) # Intervalo dinámico
        for i in range(0, len(avg_learning_curve), interval):
             if i + interval > len(avg_learning_curve): break
             episode_range = f"{i + 1}-{i + interval}"
             avg_len_interval = np.mean(avg_learning_curve[i:i + interval])
             avg_rew_interval = np.mean(avg_reward_curve[i:i + interval])
             print(f"{episode_range:<15}{avg_len_interval:<25.2f}{avg_rew_interval:<25.2f}")
        print("="*65)

        # Generar Gráficos
        print(f"\nGenerando gráficos para {env_id}...")
        window = max(1, min_episodes // 20) # Ventana de suavizado dinámica
        avg_lengths_smooth = pd.Series(avg_learning_curve).rolling(window, min_periods=1).mean()
        avg_rewards_smooth = pd.Series(avg_reward_curve).rolling(window, min_periods=1).mean()
        
        # Gráfico de Largo
        plt.figure(figsize=(12, 7))
        plt.plot(avg_learning_curve, label='Promedio por episodio', alpha=0.3)
        plt.plot(avg_lengths_smooth, label=f'Media móvil (ventana={window})', color='blue')
        plt.title(f'Curva de Aprendizaje: Largo de Episodios ({env_id} - IS-DQN)')
        plt.xlabel('Episodios')
        plt.ylabel('Largo Promedio de Episodio')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{env_id}_IS-dqn_grafico_largo_episodios.png")
        print(f"Gráfico '{env_id}_IS-dqn_grafico_largo_episodios.png' guardado.")
        
        # Gráfico de Recompensa
        plt.figure(figsize=(12, 7))
        plt.plot(avg_reward_curve, label='Promedio por episodio', alpha=0.3)
        plt.plot(avg_rewards_smooth, label=f'Media móvil (ventana={window})', color='green')
        plt.title(f'Curva de Aprendizaje: Recompensa de Episodios ({env_id} - DQN)')
        plt.xlabel('Episodios')
        plt.ylabel('Recompensa Promedio de Episodio')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{env_id}_IS-dqn_grafico_recompensa_episodios.png")
        print(f"Gráfico '{env_id}_IS-dqn_grafico_recompensa_episodios.png' guardado.")

        print(f"\nGráficos para {env_id} generados.")
        
        # --- Visualización ---
        if 1 in model_paths: # Visualizar el primer run
            print(f"\n--- Iniciando Verificación Visual del Run 1 para {env_id} ---")
            show_dqn_policy(env_id, env_config, model_paths[1], wrapper_to_apply, n_episodes=3) ### <-- CAMBIO
        elif model_paths: # O el primero que exista
            first_run_idx = sorted(model_paths.keys())[0]
            print(f"\n--- Iniciando Verificación Visual del Run {first_run_idx} para {env_id} ---")
            show_dqn_policy(env_id, env_config, model_paths[first_run_idx], wrapper_to_apply, n_episodes=3) ### <-- CAMBIO
        else:
            print(f"\nNo se encontraron modelos válidos para la verificación visual de {env_id}.")

    print(f"\n{'='*80}")
    print("🎉 Todos los experimentos han finalizado.")
    print("Mostrando todos los gráficos generados...")
    plt.show() # Mostrar todos los gráficos al final