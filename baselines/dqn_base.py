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
import sys
from stable_baselines3.common.callbacks import BaseCallback

# Agregar el directorio padre al path para que importe los módulos del proyecto
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# --- 1. IMPORTAR ENTORNOS PERSONALIZADOS ---
# (Asegúrate de que estos archivos .py estén en la misma carpeta)
from Entornos.PODoorEnv import POKeyDoorEnv
from Entornos.KeyDoorMazeEnv import KeyDoorMazeEnv
from Entornos.TwoTigersEnv import TwoTigersEnv
from Entornos.DelayedObsEnv import DelayedObsEnv
from Entornos.StochasticDelayedObsEnv import DelayedStochasticObsEnv

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


def run_dqn_experiment(env_id, env_config, h_params, run_idx):
    """
    Ejecuta una corrida de entrenamiento de DQN para un entorno dado.
    MODIFICADO: Acepta config. del entorno y guarda logs/modelos en carpetas específicas.

    :param env_id: String con el nombre del entorno (para carpetas).
    :param env_config: Diccionario con "class", "policy", "init_params".
    :param h_params: Diccionario con hiperparámetros de DQN.
    :param run_idx: El índice de la corrida actual.
    :returns: Ruta al modelo guardado o None si falla.
    """
    baselines_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(baselines_dir, f"dqn_logs/{env_id}/run_{run_idx}")
    os.makedirs(log_dir, exist_ok=True)
    model_path = os.path.join(log_dir, "dqn_model.zip")

    train_params = env_config["init_params"].copy()
    train_params["render_mode"] = "ansi"
    
    def make_env():
        env = env_config["class"](**train_params)
        env = Monitor(env, filename=log_dir)
        return env

    env = DummyVecEnv([make_env])

    # Definir el modelo DQN
    model = DQN(
        env_config["policy"],
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


def show_dqn_policy(env_id, env_config, model_path, n_episodes=3):
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
        env = env_config["class"](**vis_params)
        
        # Envolver en DummyVecEnv para que coincida con la forma de entrenamiento
        env = DummyVecEnv([lambda: env])

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
        #     "init_params": {"height": 15, "width": 19, "max_episode_steps": 200}
        # },
        "DelayedStochasticObsEnv": {
            "class": DelayedStochasticObsEnv,
            "policy": "MlpPolicy", # 3x3 grid será aplanado
            "init_params": {"size": 10, "delay_steps": 3, "max_episode_steps": 1000}
        },
        # "TwoTigersEnv": {
        #     "class": TwoTigersEnv,
        #     "policy": "MlpPolicy", # MultiDiscrete obs será aplanado y one-hot
        #     "init_params": {"max_episode_steps": 1000}
        # },
        # "DelayedObsEnv": {
        #     "class": DelayedObsEnv,
        #     "policy": "MlpPolicy", # 3x3 grid será aplanado
        #     "init_params": {"size": 10, "delay_steps": 3, "max_episode_steps": 1000}
        # }
    }

    # --- 3. DEFINIR HIPERPARÁMETROS ---
    # (Estos son genéricos, pueden necesitar ajuste por entorno)
    final_params = {
        'num_runs':10,
        'total_timesteps': 6_500_000,
        'learning_rate': 2.5e-5,     
        'buffer_size': 100_000,        
        'learning_starts': 20_000,     
        'batch_size': 2048,           
        'gamma': 0.95,               
        'train_freq': (4, "step"),   
        #'target_update_interval': 2500, 
        'tau': 0.005,                
        'target_update_interval': 1,
        'exploration_fraction': 0.2,   
        'exploration_final_eps': 0.001, 
        'policy_kwargs': dict(net_arch=[64, 64]) 
    }

    # --- 4. BUCLE PRINCIPAL DE ENTORNOS ---
    for env_id, env_config in ENVIRONMENTS.items():
        
        print(f"\n{'='*80}")
        print(f"🚀 Iniciando Experimento para Entorno: {env_id}")
        print(f"{'='*80}")

        # Rutas de logs y modelos específicas del entorno
        baselines_dir = os.path.dirname(os.path.abspath(__file__))
        base_log_dir = os.path.join(baselines_dir, f"dqn_logs/{env_id}")
        log_files_pattern = f"{base_log_dir}/run_*/monitor.csv"
        model_paths = {} # Resetear para cada entorno

        print(f"Iniciando experimento con DQN para {env_id}...")

        # --- Comprobar si los resultados ya existen ---
        existing_log_files = glob.glob(log_files_pattern)
        valid_log_count = 0
        
        for i in range(1, final_params['num_runs'] + 1):
            f = f"{base_log_dir}/run_{i}/monitor.csv"
            m = f"{base_log_dir}/run_{i}/dqn_model.zip"
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
        plt.title(f'Curva de Aprendizaje: Largo de Episodios ({env_id} - DQN)')
        plt.xlabel('Episodios')
        plt.ylabel('Largo Promedio de Episodio')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{env_id}_dqn_grafico_largo_episodios.png")
        print(f"Gráfico '{env_id}_dqn_grafico_largo_episodios.png' guardado.")
        
        # Gráfico de Recompensa
        plt.figure(figsize=(12, 7))
        plt.plot(avg_reward_curve, label='Promedio por episodio', alpha=0.3)
        plt.plot(avg_rewards_smooth, label=f'Media móvil (ventana={window})', color='green')
        plt.title(f'Curva de Aprendizaje: Recompensa de Episodios ({env_id} - DQN)')
        plt.xlabel('Episodios')
        plt.ylabel('Recompensa Promedio de Episodio')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{env_id}_dqn_grafico_recompensa_episodios.png")
        print(f"Gráfico '{env_id}_dqn_grafico_recompensa_episodios.png' guardado.")

        print(f"\nGráficos para {env_id} generados.")
        
        # --- Visualización ---
        if 1 in model_paths: # Visualizar el primer run
            print(f"\n--- Iniciando Verificación Visual del Run 1 para {env_id} ---")
            show_dqn_policy(env_id, env_config, model_paths[1], n_episodes=3)
        elif model_paths: # O el primero que exista
            first_run_idx = sorted(model_paths.keys())[0]
            print(f"\n--- Iniciando Verificación Visual del Run {first_run_idx} para {env_id} ---")
            show_dqn_policy(env_id, env_config, model_paths[first_run_idx], n_episodes=3)
        else:
            print(f"\nNo se encontraron modelos válidos para la verificación visual de {env_id}.")

    print(f"\n{'='*80}")
    print("🎉 Todos los experimentos han finalizado.")
    print("Mostrando todos los gráficos generados...")
    plt.show() # Mostrar todos los gráficos al final