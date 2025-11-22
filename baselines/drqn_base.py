import gymnasium as gym
import numpy as np
import pandas as pd
import time
import os
import glob
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import collections
import random
import sys
from stable_baselines3.common.monitor import Monitor # Para crear 'monitor.csv'

# Agregar el directorio padre al path para que importe los módulos del proyecto
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Entornos.PODoorEnv import POKeyDoorEnv
from Entornos.KeyDoorMazeEnv import KeyDoorMazeEnv
from Entornos.TwoTigersEnv import TwoTigersEnv
from Entornos.DelayedObsEnv import DelayedObsEnv

# --- INICIO: LÓGICA CENTRAL DE DRQN (DE KEEP9OING) ---

def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def save_model(model, path='default.pth'):
    # El ejemplo de SB3 guarda el .zip, nosotros guardamos el .pth
    torch.save(model.state_dict(), path)

# Red Q (MLP + LSTM) para entornos tabulares/vectoriales
class Q_net(nn.Module):
    def __init__(self, state_space, action_space):
        super(Q_net, self).__init__()

        self.hidden_space = 64 # Puedes ajustar esto
        self.state_space = state_space
        self.action_space = action_space

        self.Linear1 = nn.Linear(self.state_space, self.hidden_space)
        self.lstm = nn.LSTM(self.hidden_space, self.hidden_space, batch_first=True)
        self.Linear2 = nn.Linear(self.hidden_space, self.action_space)

    # def forward(self, x, h, c):
    #     x = F.relu(self.Linear1(x))
    #     # Asegurarnos de que la entrada al LSTM sea (batch, seq_len, features)
    #     x, (new_h, new_c) = self.lstm(x, (h, c))
    #     x = self.Linear2(x)
    #     return x, new_h, new_c
    
    def forward(self, x, h, c):
        # x entra con shape (batch_size, seq_len, 3, 3) o (batch_size, seq_len, N)
        # Lo aplanamos para que sea (batch_size, seq_len, features)
        x = x.view(x.size(0), x.size(1), -1) # <--- ¡AÑADE ESTA LÍNEA!

        x = F.relu(self.Linear1(x)) # (batch, seq, 9) -> (batch, seq, 64)
        x, (new_h, new_c) = self.lstm(x,(h,c))
        x = self.Linear2(x)
        return x, new_h, new_c

    def sample_action(self, obs, h, c, epsilon):
        # El obs que entra es (features,)
        # Lo convertimos a (1, 1, features) para el LSTM
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).unsqueeze(0).to(h.device)
        output = self.forward(obs_tensor, h, c)

        if random.random() < epsilon:
            # .item() para CPU, .cpu().item() si 'random.randint' está en CPU
            return random.randint(0, self.action_space - 1), output[1], output[2]
        else:
            return output[0].argmax().item(), output[1], output[2]

    def init_hidden_state(self, batch_size, device, training=None):
        assert training is not None, "training step parameter should be dtermined"
        if training is True:
            return (
                torch.zeros([1, batch_size, self.hidden_space]).to(device),
                torch.zeros([1, batch_size, self.hidden_space]).to(device),
            )
        else:
            return (
                torch.zeros([1, 1, self.hidden_space]).to(device),
                torch.zeros([1, 1, self.hidden_space]).to(device),
            )

# Buffer de Memoria (almacena episodios completos)
class EpisodeMemory:
    def __init__(self, random_update=False, max_epi_num=100, batch_size=1, lookup_step=None):
        self.random_update = random_update
        self.max_epi_num = max_epi_num
        self.batch_size = batch_size
        self.lookup_step = lookup_step
        self.memory = collections.deque(maxlen=self.max_epi_num)

    def put(self, episode):
        self.memory.append(episode)

    def sample(self):
        sampled_buffer = []
        if self.random_update:
            if len(self.memory) < self.batch_size:
                return None, None # No hay suficientes episodios para muestrear
            
            sampled_episodes = random.sample(self.memory, self.batch_size)
            min_step = float('inf')
            for episode in sampled_episodes:
                min_step = min(min_step, len(episode))

            for episode in sampled_episodes:
                if min_step > self.lookup_step:
                    idx = np.random.randint(0, len(episode) - self.lookup_step + 1)
                    sample = episode.sample(random_update=True, lookup_step=self.lookup_step, idx=idx)
                    sampled_buffer.append(sample)
                else:
                    # Si el episodio más corto es más corto que el lookup, usamos el episodio más corto
                    idx = np.random.randint(0, len(episode) - min_step + 1)
                    sample = episode.sample(random_update=True, lookup_step=min_step, idx=idx)
                    sampled_buffer.append(sample)
            
            return sampled_buffer, len(sampled_buffer[0]['obs'])
        else:
            # Muestreo secuencial (no lo usaremos, pero se queda por completitud)
            idx = np.random.randint(0, len(self.memory))
            sampled_buffer.append(self.memory[idx].sample(random_update=False))
            return sampled_buffer, len(sampled_buffer[0]['obs'])

    def __len__(self):
        return len(self.memory)

# Buffer de Episodio (almacena transiciones de un solo episodio)
class EpisodeBuffer:
    def __init__(self):
        self.obs = []
        self.action = []
        self.reward = []
        self.next_obs = []
        self.done = []

    def put(self, transition):
        self.obs.append(transition[0])
        self.action.append(transition[1])
        self.reward.append(transition[2])
        self.next_obs.append(transition[3])
        self.done.append(transition[4])

    def sample(self, random_update=False, lookup_step=None, idx=None) -> dict[str, np.ndarray]:
        obs = np.array(self.obs)
        action = np.array(self.action)
        reward = np.array(self.reward)
        next_obs = np.array(self.next_obs)
        done = np.array(self.done)

        if random_update is True:
            obs = obs[idx : idx + lookup_step]
            action = action[idx : idx + lookup_step]
            reward = reward[idx : idx + lookup_step]
            next_obs = next_obs[idx : idx + lookup_step]
            done = done[idx : idx + lookup_step]

        return dict(obs=obs, acts=action, rews=reward, next_obs=next_obs, done=done)

    def __len__(self) -> int:
        return len(self.obs)

# Función de Entrenamiento
def train(q_net, target_q_net, episode_memory, device, optimizer, batch_size, gamma):
    
    samples, seq_len = episode_memory.sample()
    if samples is None:
        return # No entrenar si no hay suficientes datos

    observations, actions, rewards, next_observations, dones = [], [], [], [], []

    for i in range(batch_size):
        observations.append(samples[i]["obs"])
        actions.append(samples[i]["acts"])
        rewards.append(samples[i]["rews"])
        next_observations.append(samples[i]["next_obs"])
        dones.append(samples[i]["done"])

    # Convertir a arrays y luego a tensores
    observations = np.array(observations)
    actions = np.array(actions)
    rewards = np.array(rewards)
    next_observations = np.array(next_observations)
    dones = np.array(dones)

    # Asegurarse de que el shape sea (batch_size, seq_len, features)
    obs_shape = observations.shape
    if len(obs_shape) == 2: # (batch, seq) -> (batch, seq, 1) para estados tabulares simples
         observations = np.expand_dims(observations, -1)
         next_observations = np.expand_dims(next_observations, -1)
    
    # Redimensionar para el LSTM: (batch_size, seq_len, features)
    observations = torch.FloatTensor(observations).to(device)
    actions = torch.LongTensor(actions).unsqueeze(-1).to(device) # (batch, seq, 1)
    rewards = torch.FloatTensor(rewards).unsqueeze(-1).to(device) # (batch, seq, 1)
    next_observations = torch.FloatTensor(next_observations).to(device)
    dones = torch.FloatTensor(dones).unsqueeze(-1).to(device) # (batch, seq, 1)

    h_target, c_target = target_q_net.init_hidden_state(batch_size=batch_size, device=device, training=True)
    q_target, _, _ = target_q_net(next_observations, h_target, c_target)
    q_target_max = q_target.max(2)[0].unsqueeze(-1).detach() # (batch, seq, 1)
    targets = rewards + gamma * q_target_max * dones

    h, c = q_net.init_hidden_state(batch_size=batch_size, device=device, training=True)
    q_out, _, _ = q_net(observations, h, c)
    q_a = q_out.gather(2, actions)

    loss = F.smooth_l1_loss(q_a, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# --- FIN: LÓGICA CENTRAL DE DRQN ---


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
        print(f"   Dispositivos encontrados: {device_count}")
        print(f"   Usando dispositivo {current_dev_idx}: {device_name}")
        return torch.device("cuda")
    else:
        print(f"❌ GPU no disponible.")
        print("   El entrenamiento se ejecutará en la CPU.")
        return torch.device("cpu")
    print(f"{'='*40}\n")


def run_drqn_experiment(env_id, env_config, h_params, run_idx, device):
    """
    Ejecuta una corrida de entrenamiento de DRQN (PyTorch) para un entorno dado.
    """
    baselines_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(baselines_dir, f"drqn_logs/{env_id}/run_{run_idx}")
    os.makedirs(log_dir, exist_ok=True)
    model_path = os.path.join(log_dir, "drqn_model.pth") # Guardamos como .pth

    # --- 1. Crear el Entorno ---
    def make_env():
        # Usa la clase y los parámetros de la configuraciónt
        env = env_config["class"](**env_config["init_params"])
        # ¡IMPORTANTE! Envolvemos con Monitor para guardar 'monitor.csv'
        env = Monitor(env, filename=log_dir)
        return env

    env = make_env()
    
    # --- 2. Setear Semillas ---
    seed = run_idx # Usar el índice de corrida como semilla
    np.random.seed(seed)
    random.seed(seed)
    seed_torch(seed)
    # env.seed(seed) # Monitor no tiene .seed, se setea en reset

    # --- 3. Inicializar Modelo, Optimizador y Memoria ---
    
    # Generalizar para inputs tabulares/vectoriales
    if isinstance(env.observation_space, gym.spaces.Discrete):
        # Si el estado es un solo número, asumimos one-hot o embedding
        # Por ahora, usaremos el número de estados como input_size
        # ¡MEJORA! Deberías usar un wrapper para convertir esto a one-hot
        print("Advertencia: Entorno discreto, usando 'n' como state_space.")
        state_size = env.observation_space.n
    else:
        # Para Box (vectores), aplanamos
        state_size = np.prod(env.observation_space.shape)
        
    action_size = env.action_space.n

    Q = Q_net(state_space=state_size, action_space=action_size).to(device)
    Q_target = Q_net(state_space=state_size, action_space=action_size).to(device)
    Q_target.load_state_dict(Q.state_dict())
    optimizer = optim.Adam(Q.parameters(), lr=h_params['learning_rate'])

    episode_memory = EpisodeMemory(
        random_update=h_params['random_update'],
        max_epi_num=h_params['max_epi_num'],
        batch_size=h_params['batch_size'],
        lookup_step=h_params['lookup_step']
    )

    epsilon = h_params['eps_start']
    score_sum = 0
    
    print(f"    Iniciando entrenamiento para {env_id} - corrida {run_idx}...")
    
    try:
        # --- 4. Bucle de Entrenamiento (adaptado de DRQN.py) ---
        for i in range(h_params['episodes']):
            # Usar 'seed' en reset() para la API moderna de gymnasium
            s, info = env.reset(seed=seed + i)
            obs = s # Asumimos que el env ya devuelve el vector correcto
            done = False
            
            episode_record = EpisodeBuffer()
            h, c = Q.init_hidden_state(batch_size=h_params['batch_size'], device=device, training=False)
            
            # El 'score' ahora lo maneja el Monitor, pero lo guardamos para el log
            current_episode_score = 0

            for t in range(h_params['max_step']):
                
                # --- Lógica de Inferencia ---
                a, h, c = Q.sample_action(obs, h, c, epsilon)
                
                # --- Lógica de Entorno ---
                s_prime, r, term, trunc, info = env.step(a)
                done = term or trunc
                obs_prime = s_prime # Asumir que s_prime es el vector correcto
                
                done_mask = 0.0 if done else 1.0
                episode_record.put([obs, a, r, obs_prime, done_mask])
                obs = obs_prime
                
                current_episode_score += r
                score_sum += r # Para el log de consola

                # --- Lógica de Entrenamiento ---
                if len(episode_memory) >= h_params['min_epi_num']:
                    train(Q, Q_target, episode_memory, device,
                          optimizer=optimizer,
                          batch_size=h_params['batch_size'],
                          gamma=h_params['gamma'])
                    
                    # Actualización de la Target Network (soft update)
                    if (t + 1) % h_params['target_update_period'] == 0:
                        for target_param, local_param in zip(Q_target.parameters(), Q.parameters()):
                            target_param.data.copy_(h_params['tau'] * local_param.data + (1.0 - h_params['tau']) * target_param.data)
                
                if done:
                    break
            
            episode_memory.put(episode_record)
            epsilon = max(h_params['eps_end'], epsilon * h_params['eps_decay'])
            
            # Imprimir progreso (similar al callback de SB3)
            if i % h_params['print_per_iter'] == 0 and i != 0:
                print(f"    [Episodio {i}] Recompensa media (últimos {h_params['print_per_iter']} ep): {score_sum/h_params['print_per_iter']:.2f}, Buffer: {len(episode_memory)}, Eps: {epsilon*100:.1f}%")
                score_sum = 0.0
                
        # --- 5. Guardar el Modelo ---
        print(f"    Entrenamiento completado para corrida {run_idx}.")
        print(f"    Guardando modelo en: {model_path}")
        save_model(Q, model_path)
        env.close()
        return model_path

    except Exception as e:
        print(f"    ERROR durante el entrenamiento para {env_id} corrida {run_idx}: {e}")
        env.close()
        return None

def show_drqn_policy(env_id, env_config, model_path, n_episodes=3, device=torch.device('cpu')):
    """
    Carga un modelo DRQN (PyTorch) guardado y lo ejecuta para visualización.
    """
    print(f"\nCargando modelo DRQN desde {model_path} para visualización...")
    if not os.path.exists(model_path):
        print(f"Error: No se encontró el archivo del modelo en {model_path}")
        return

    try:
        # --- Instanciar entorno personalizado para visualización ---
        vis_params = env_config["init_params"].copy()
        vis_params["render_mode"] = "human" # Forzar modo 'human'
        env = env_config["class"](**vis_params)
        
        # Generalizar para inputs tabulares/vectoriales
        if isinstance(env.observation_space, gym.spaces.Discrete):
            state_size = env.observation_space.n
        else:
            state_size = np.prod(env.observation_space.shape)
        action_size = env.action_space.n
        
        # Cargar el modelo
        model = Q_net(state_space=state_size, action_space=action_size).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval() # Poner en modo evaluación

        for episode in range(n_episodes):
            obs, info = env.reset()
            done = False
            total_reward = 0
            steps = 0
            
            # Inicializar estado oculto para inferencia
            h, c = model.init_hidden_state(batch_size=1, device=device, training=False)
            print(f"\nIniciando visualización - Episodio {episode + 1}/{n_episodes}")
            
            while not done:
                env.render()
                
                # Usar sample_action con epsilon 0 para política determinista
                action, h, c = model.sample_action(obs, h, c, epsilon=0.0)
                
                obs, reward, term, trunc, info = env.step(action)
                done = term or trunc
                
                total_reward += reward
                steps += 1
                
                if done:
                    print(f"Episodio {episode + 1} terminado en {steps} pasos. Recompensa total: {total_reward:.2f}")

        env.close()
        print("\nVisualización completada.")

    except Exception as e:
        print(f"Error al cargar o ejecutar el modelo DRQN para visualización: {e}")


if __name__ == "__main__":
    
    device = check_gpu() # Verificar GPU antes de comenzar

    # --- 2. DEFINIR LOS ENTORNOS A PROBAR ---
    # (¡Asegúrate de que tus entornos estén importados arriba!)
    # (¡Asegúrate de que tus entornos devuelvan un vector de estado, no una imagen!)
    ENVIRONMENTS = {
        # "POKeyDoorEnv": {
        #     "class": POKeyDoorEnv,
        #     "init_params": {"size": 10, "max_episode_steps": 1000}
        # },
        "KeyDoorMazeEnv": {
            "class": KeyDoorMazeEnv,
            "init_params": {"height": 15, "width": 19, "max_episode_steps": 200}
        },
        # "TwoTigersEnv": {
        #     "class": TwoTigersEnv,
        #     "init_params": {"max_episode_steps": 50}
        # },
        # "DelayedObsEnv": {
        #     "class": DelayedObsEnv,
        #     "init_params": {"size": 10, "delay_steps": 3, "max_episode_steps": 1000}
        # },
        # "CartPole_POMDP": {
        #     "class": gym.make, # Usamos el 'make' de gym como una 'clase'
        #     "init_params": {"id": "CartPole-v1", "max_episode_steps": 1000}
        # }
    }

    # --- 3. DEFINIR HIPERPARÁMETROS PARA DRQN ---
    # (Basados en el script DRQN.py)
    final_params = {
        'num_runs': 10, # 10 corridas
        'episodes': 1500, # Timesteps total = episodes * max_step
        'max_step': 1000, # Máximos pasos por episodio
        'learning_rate': 2.5e-4,
        'gamma': 0.95,
        'batch_size': 8, 
        'buffer_size': 100_000, # (No usado por EpisodeMemory, pero lo guardamos)
        'min_epi_num': 20, # Empezar a entrenar después de 20 episodios
        'target_update_period': 4,
        'eps_start': 0.2,
        'eps_end': 0.001,
        'eps_decay': 0.995,
        'tau': 0.005, # Para soft update
        'random_update': True,
        'lookup_step': 20,
        'max_epi_num': 100, # Tamaño del buffer (en episodios)
        'print_per_iter': 20 # Frecuencia de log
    }

    # --- 4. BUCLE PRINCIPAL DE ENTORNOS ---
    for env_id, env_config in ENVIRONMENTS.items():
        
        print(f"\n{'='*80}")
        print(f"🚀 Iniciando Experimento para Entorno: {env_id} (DRQN Baseline)")
        print(f"{'='*80}")

        baselines_dir = os.path.dirname(os.path.abspath(__file__))
        base_log_dir = os.path.join(baselines_dir, f"drqn_logs/{env_id}")
        log_files_pattern = f"{base_log_dir}/run_*/monitor.csv"
        model_paths = {}

        # --- Comprobar si los resultados ya existen ---
        valid_log_count = 0
        for i in range(1, final_params['num_runs'] + 1):
            f = f"{base_log_dir}/run_{i}/monitor.csv"
            m = f"{base_log_dir}/run_{i}/drqn_model.pth" # <-- Cambiado a .pth
            if os.path.exists(f) and os.path.exists(m):
                 try:
                     pd.read_csv(f, skiprows=1, nrows=1)
                     valid_log_count += 1
                     model_paths[i] = m 
                 except (pd.errors.EmptyDataError, FileNotFoundError):
                     continue 

        if valid_log_count >= final_params['num_runs']:
            print(f"Se encontraron {valid_log_count} archivos de log y modelos existentes para {env_id}. Saltando la fase de entrenamiento.")
        else:
            print(f"Se encontraron {valid_log_count} archivos válidos para {env_id} (se necesitan {final_params['num_runs']}). Iniciando entrenamiento...")
            overall_start_time = time.time()

            for i in range(final_params['num_runs']):
                run_idx = i + 1
                specific_log_path = f"{base_log_dir}/run_{run_idx}/monitor.csv"
                specific_model_path = f"{base_log_dir}/run_{run_idx}/drqn_model.pth" # <-- Cambiado a .pth
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
                    
                    saved_model_path = run_drqn_experiment(
                        env_id, env_config, final_params, run_idx, device
                    )
                    
                    if saved_model_path:
                        model_paths[run_idx] = saved_model_path
                    run_time = time.time() - run_start_time
                    print(f"Corrida {run_idx} completada en {run_time:.2f} segundos.")

            print(f"\nEntrenamiento de las {final_params['num_runs']} corridas completado para {env_id}.")
            total_training_time = (time.time() - overall_start_time)
            print(f"Tiempo total de entrenamiento para {env_id}: {total_training_time/60:.2f} minutos.")


        # --- 5. Procesar y Reportar Resultados ---
        # (Esta sección es idéntica a tu script de SB3, leerá los 'monitor.csv')
        print(f"\nProcesando resultados para {env_id}...")

        if not model_paths:
            print(f"No se encontraron archivos de log o modelos válidos para {env_id}. Saltando reporte.")
            continue
            
        all_episode_lengths = []
        all_episode_rewards = []
        min_episodes = float('inf')
        valid_files_processed = 0

        for run_key in sorted(model_paths.keys()):
             f = f"{base_log_dir}/run_{run_key}/monitor.csv"
             if os.path.exists(f):
                 try:
                     # El Monitor de SB3 guarda 2 líneas de cabecera
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
            continue
        
        print(f"Procesando {valid_files_processed} logs válidos para {env_id}.")
        print(f"Truncando a {min_episodes} episodios.")

        run_key_to_list_index = {key: idx for idx, key in enumerate(sorted(model_paths.keys()))}
        
        padded_lengths = []
        padded_rewards = []
        for k in sorted(model_paths.keys()):
            # Chequear si la llave existe en el mapeo (si el log era válido)
            if k in run_key_to_list_index:
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
        print(f" " * 10 + f"REPORTE PROMEDIO DE EPISODIOS (DRQN - {env_id})")
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
        plt.plot(avg_learning_curve, label='Promedio por episodio', alpha=0.3, color='blue')
        plt.plot(avg_lengths_smooth, label=f'Media móvil (ventana={window})', color='blue')
        plt.title(f'Curva de Aprendizaje: Largo de Episodios ({env_id} - DRQN)')
        plt.xlabel('Episodios')
        plt.ylabel('Largo Promedio de Episodio')
        plt.legend()
        plt.grid(True)
        # Guardar en la carpeta de logs del entorno
        graph_path_largo = os.path.join(base_log_dir, f"{env_id}_drqn_grafico_largo.png")
        plt.savefig(graph_path_largo)
        print(f"Gráfico '{graph_path_largo}' guardado.")
        
        
        plt.figure(figsize=(12, 7))
        plt.plot(avg_reward_curve, label='Promedio por episodio', alpha=0.3)
        plt.plot(avg_rewards_smooth, label=f'Media móvil (ventana={window})', color='green')
        plt.title(f'Curva de Aprendizaje: Recompensa de Episodios ({env_id} - DRQN)')
        plt.xlabel('Episodios')
        plt.ylabel('Recompensa Promedio de Episodio')
        plt.legend()
        plt.grid(True)
        graph_path_recompensa = os.path.join(base_log_dir, f"{env_id}_drqn_grafico_recompensa.png")
        plt.savefig(graph_path_recompensa)
        print(f"Gráfico '{graph_path_recompensa}' guardado.")

        print(f"\nGráficos para {env_id} generados.")
        
        # --- Visualización ---
        if 1 in model_paths:
            print(f"\n--- Iniciando Verificación Visual del Run 1 para {env_id} ---")
            show_drqn_policy(env_id, env_config, model_paths[1], n_episodes=3, device=device)
        elif model_paths:
            first_run_idx = sorted(model_paths.keys())[0]
            print(f"\n--- Iniciando Verificación Visual del Run {first_run_idx} para {env_id} ---")
            show_drqn_policy(env_id, env_config, model_paths[first_run_idx], n_episodes=3, device=device)

    print(f"\n{'='*80}")
    print("🎉 Todos los experimentos han finalizado.")
    print("Mostrando todos los gráficos generados...")
    plt.show()