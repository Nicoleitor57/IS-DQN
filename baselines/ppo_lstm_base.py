import gymnasium as gym
import numpy as np
import pandas as pd
import time
import os
import glob
import matplotlib.pyplot as plt
import torch 
import sys
from stable_baselines3.common.callbacks import BaseCallback


from sb3_contrib import RecurrentPPO  
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

# Agregar el directorio padre al path para que importe los módulos del proyecto
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Importar Entornos Personalizados ---
from Entornos.PODoorEnv import POKeyDoorEnv
from Entornos.KeyDoorMazeEnv import KeyDoorMazeEnv
from Entornos.TwoTigersEnv import TwoTigersEnv
from Entornos.DelayedObsEnv import DelayedObsEnv


def check_gpu():
    """Verifica si PyTorch (backend de SB3) puede detectar y usar la GPU."""
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
    """Callback simple que imprime la recompensa media de los últimos 10 episodios."""
    def __init__(self, check_freq: int, verbose: int = 1):
        super(SimpleProgressCallback, self).__init__(verbose)
        self.check_freq = check_freq

    def _on_step(self) -> bool:
        if self.num_timesteps % self.check_freq == 0:
            monitor = self.training_env.envs[0]
            recent_rewards = monitor.get_episode_rewards()[-10:]
            if recent_rewards:
                mean_reward = np.mean(recent_rewards)
                print(f"    [Paso {self.num_timesteps}] Recompensa media (últimos 10 ep): {mean_reward:.2f}")
        return True


def run_ppo_experiment(env_id, env_config, h_params, run_idx):
    """
    Ejecuta una corrida de entrenamiento de PPO Recurrente.
    """
    # Directorio de logs
    baselines_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(baselines_dir, f"ppo_logs/{env_id}/run_{run_idx}") # Carpeta PPO
    os.makedirs(log_dir, exist_ok=True)
    model_path = os.path.join(log_dir, "ppo_model.zip") # Modelo PPO

    # Instanciar entorno
    train_params = env_config["init_params"].copy()
    train_params["render_mode"] = "ansi"
    
    def make_env():
        env = env_config["class"](**train_params)
        env = Monitor(env, filename=log_dir)
        return env

    env = DummyVecEnv([make_env])

    # Definir el modelo PPO con política Recurrente (LSTM)
    model = RecurrentPPO(
        env_config["policy"], # Usar la política definida (ej. "MlpLstmPolicy")
        env,
        learning_rate=h_params['learning_rate'],
        n_steps=h_params['n_steps'],
        batch_size=h_params['batch_size'],
        n_epochs=h_params['n_epochs'],
        gamma=h_params['gamma'],
        gae_lambda=h_params['gae_lambda'],
        clip_range=h_params['clip_range'],
        ent_coef=h_params['ent_coef'],
        policy_kwargs=h_params['policy_kwargs'],
        verbose=0,
        device="auto" 
    )
    
    print(f"    Modelo PPO creado. Dispositivo en uso: {model.device}")
    
    progress_callback = SimpleProgressCallback(check_freq=5000)

    print(f"    Iniciando entrenamiento para {env_id} - corrida {run_idx}...")
    try:
        model.learn(
            total_timesteps=h_params['total_timesteps'],
            callback=progress_callback 
        )
        
        print(f"    Entrenamiento completado para corrida {run_idx}.")
        print(f"    Guardando modelo en: {model_path}")
        model.save(model_path)
        env.close()
        return model_path
    except Exception as e:
        print(f"    ERROR durante el entrenamiento para {env_id} corrida {run_idx}: {e}")
        env.close()
        return None

def show_ppo_policy(env_id, env_config, model_path, n_episodes=3):
    """
    Carga un modelo PPO guardado y lo ejecuta para visualización.
    """
    print(f"\nCargando modelo PPO desde {model_path} para visualización...")
    if not os.path.exists(model_path):
        print(f"Error: No se encontró el archivo del modelo en {model_path}")
        return

    try:
        # Cargar el modelo PPO
        model = RecurrentPPO.load(model_path)
        
        vis_params = env_config["init_params"].copy()
        vis_params["render_mode"] = "human"
        
        # --- Manejo de estado recurrente para visualización ---
        # Creamos el env sin DummyVecEnv para un control manual del estado
        env = env_config["class"](**vis_params)

        for episode in range(n_episodes):
            obs, _ = env.reset()
            # PPO-LSTM necesita que el estado se reinicie manualmente
            lstm_states = None 
            done = False
            total_reward = 0
            steps = 0
            print(f"\nIniciando visualización - Episodio {episode + 1}/{n_episodes}")
            
            while not done:
                # Pasar el estado LSTM anterior y obtener el nuevo
                action, lstm_states = model.predict(
                    obs, 
                    state=lstm_states, 
                    deterministic=True
                )
                
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_reward += reward
                steps += 1
                
                # Si el episodio termina, el estado LSTM se reiniciará automáticamente
                # en el próximo env.reset(), pero es bueno saberlo.

            print(f"Episodio {episode + 1} terminado en {steps} pasos. Recompensa total: {total_reward:.2f}")

        env.close()
        print("\nVisualización completada.")

    except Exception as e:
        print(f"Error al cargar o ejecutar el modelo PPO para visualización: {e}")


if __name__ == "__main__":

    check_gpu()

    # --- 2. DEFINIR LOS ENTORNOS A PROBAR ---
    ENVIRONMENTS = {
        # "POKeyDoorEnv": {
        #     "class": POKeyDoorEnv,
        #     "policy": "MlpLstmPolicy", 
        #     "init_params": {"size": 10, "max_episode_steps": 1000}
        # },
        #"KeyDoorMazeEnv": {
        #    "class": KeyDoorMazeEnv,
        #   "policy": "MlpLstmPolicy",
        #    "init_params": {"height": 15, "width": 19, "max_episode_steps": 200}
        #},
         "TwoTigersEnv": {
             "class": TwoTigersEnv,
             "policy": "MlpLstmPolicy", 
             "init_params": {"max_episode_steps": 50}
         },
        # "DelayedObsEnv": {
        #     "class": DelayedObsEnv,
        #     "policy": "MlpLstmPolicy", 
        #     "init_params": {"size": 10, "delay_steps": 3, "max_episode_steps": 1000}
        # }
    }

    
    final_params = {
    'num_runs': 10,
    'total_timesteps': 300_000,
    'learning_rate': 2.5e-5,       # Tasa de aprendizaje
    'n_steps': 256,              # Pasos por recolección antes de la actualización
    'batch_size': 32,            # Tamaño del minibatch
    'n_epochs': 10,              # Épocas de optimización por actualización
    'gamma': 0.95,               # Factor de descuento
    'gae_lambda': 0.95,          # Factor GAE
    'clip_range': 0.2,           # Rango de PPO clip
    'ent_coef': 0.0,             # Coeficiente de entropía (regularización)
     # Parámetros específicos de la política LSTM
    'policy_kwargs': dict(
         net_arch=dict(pi=[64, 64], vf=[64, 64]), # Arquitectura
         lstm_hidden_size=64, # Tamaño del estado oculto
         enable_critic_lstm=True, # El crítico también usa LSTM
     ) 
}    
# final_params = {
# 'num_runs': 10,
# 'total_timesteps': 300_000,
# 'learning_rate': 1e-4,       # Ligeramente más alta para más velocidad inicial
# 'n_steps': 1024,             # <--- AUMENTADO: Horizonte de memoria
# 'batch_size': 128,           # <--- AUMENTADO: Estabilidad con n_steps más largo
# 'n_epochs': 10,
# 'gamma': 0.99,               # <--- AUMENTADO: Valorar mucho más el futuro (Puerta al final)
# 'gae_lambda': 0.95,
# 'clip_range': 0.2,
# 'ent_coef': 0.01,            # <--- AUMENTADO: Más exploración en laberinto
#'policy_kwargs': dict(
#     net_arch=dict(pi=[128, 128], vf=[128, 128]), # <--- AUMENTADO: Mayor capacidad
#      lstm_hidden_size=128,                        # <--- AUMENTADO: Más memoria
#       enable_critic_lstm=True,
#     
#}

    # --- 4. BUCLE PRINCIPAL DE ENTORNOS ---
    for env_id, env_config in ENVIRONMENTS.items():
        
        print(f"\n{'='*80}")
        print(f"🚀 Iniciando Experimento Recurrente para Entorno: {env_id}")
        print(f"{'='*80}")

        # Rutas de logs y modelos
        baselines_dir = os.path.dirname(os.path.abspath(__file__))
        base_log_dir = os.path.join(baselines_dir, f"ppo_logs/{env_id}") ### CAMBIO ###
        log_files_pattern = f"{base_log_dir}/run_*/monitor.csv"
        model_paths = {}

        print(f"Iniciando experimento con PPO-LSTM para {env_id}...")

        # --- Comprobar si los resultados ya existen ---
        # (Esta lógica es la misma, solo cambian las rutas)
        valid_log_count = 0
        for i in range(1, final_params['num_runs'] + 1):
            f = f"{base_log_dir}/run_{i}/monitor.csv"
            m = f"{base_log_dir}/run_{i}/ppo_model.zip" ### CAMBIO ###
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
            overall_start_time = time.time()
            all_run_times = []

            # --- Ejecutar los 30 experimentos ---
            for i in range(final_params['num_runs']):
                run_idx = i + 1
                specific_log_path = f"{base_log_dir}/run_{run_idx}/monitor.csv"
                specific_model_path = f"{base_log_dir}/run_{run_idx}/ppo_model.zip" ### CAMBIO ###
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
                    
                    ### CAMBIO ###
                    saved_model_path = run_ppo_experiment(
                        env_id, env_config, final_params, run_idx
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
        # (Esta sección es idéntica, solo cambian los nombres de los gráficos)
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
                     df = pd.read_csv(f, skiprows=1)
                     if df.empty: continue
                     all_episode_lengths.append(df['l'].values)
                     all_episode_rewards.append(df['r'].values)
                     valid_files_processed += 1
                     min_episodes = min(min_episodes, len(df['l']))
                 except (pd.errors.EmptyDataError, FileNotFoundError):
                     continue
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

        print("\n" + "="*65)
        print(f" " * 10 + f"REPORTE PROMEDIO DE EPISODIOS (PPO-LSTM - {env_id})") ### CAMBIO ###
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

        print(f"\nGenerando gráficos para {env_id}...")
        window = max(1, min_episodes // 20)
        avg_lengths_smooth = pd.Series(avg_learning_curve).rolling(window, min_periods=1).mean()
        avg_rewards_smooth = pd.Series(avg_reward_curve).rolling(window, min_periods=1).mean()
        
        plt.figure(figsize=(12, 7))
        plt.plot(avg_learning_curve, label='Promedio por episodio', alpha=0.3)
        plt.plot(avg_lengths_smooth, label=f'Media móvil (ventana={window})', color='blue')
        plt.title(f'Curva de Aprendizaje: Largo de Episodios ({env_id} - PPO-LSTM)') ### CAMBIO ###
        plt.xlabel('Episodios')
        plt.ylabel('Largo Promedio de Episodio')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{env_id}_ppo_grafico_largo_episodios.png") ### CAMBIO ###
        print(f"Gráfico '{env_id}_ppo_grafico_largo_episodios.png' guardado.")
        
        plt.figure(figsize=(12, 7))
        plt.plot(avg_reward_curve, label='Promedio por episodio', alpha=0.3)
        plt.plot(avg_rewards_smooth, label=f'Media móvil (ventana={window})', color='green')
        plt.title(f'Curva de Aprendizaje: Recompensa de Episodios ({env_id} - PPO-LSTM)') ### CAMBIO ###
        plt.xlabel('Episodios')
        plt.ylabel('Recompensa Promedio de Episodio')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{env_id}_ppo_grafico_recompensa_episodios.png") ### CAMBIO ###
        print(f"Gráfico '{env_id}_ppo_grafico_recompensa_episodios.png' guardado.")
        
        print(f"\nGráficos para {env_id} generados.")
        
        if 1 in model_paths:
            print(f"\n--- Iniciando Verificación Visual del Run 1 para {env_id} ---")
            show_ppo_policy(env_id, env_config, model_paths[1], n_episodes=3) ### CAMBIO ###
        elif model_paths:
            first_run_idx = sorted(model_paths.keys())[0]
            print(f"\n--- Iniciando Verificación Visual del Run {first_run_idx} para {env_id} ---")
            show_ppo_policy(env_id, env_config, model_paths[first_run_idx], n_episodes=3) ### CAMBIO ###
        else:
            print(f"\nNo se encontraron modelos válidos para la verificación visual de {env_id}.")

    print(f"\n{'='*80}")
    print("🎉 Todos los experimentos han finalizado.")
    print("Mostrando todos los gráficos generados...")
    plt.show()
