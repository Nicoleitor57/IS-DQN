import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

def smooth_data(data, window_size):
    """
    Suaviza los datos usando una media móvil.
    """
    if data.ndim == 1:
        s = pd.Series(data)
        return s.rolling(window_size, min_periods=1).mean().values
    elif data.ndim == 2:
        smoothed_runs = []
        for run in data:
            s = pd.Series(run)
            smoothed_runs.append(s.rolling(window_size, min_periods=1).mean().values)
        return np.array(smoothed_runs)
    return None

def load_log_data(base_path, env_name, num_runs):
    """
    Carga los datos de 'monitor.csv' para un algoritmo y entorno específico.
    """
    all_rewards = []
    all_lengths = []
    min_len = float('inf') 

    target_dir = os.path.join(base_path, env_name)
    print(f"  -> Buscando logs en: {target_dir}")

    # Verificar si la carpeta del entorno existe para este algoritmo
    if not os.path.exists(target_dir):
        print(f"     Advertencia: No existe la carpeta {target_dir}")
        return None, None, 0

    for i in range(1, num_runs + 1):
        log_file = os.path.join(target_dir, f'run_{i}', 'monitor.csv')
        
        # Lógica de fallback por si las corridas no son secuenciales (ej. run_1, run_10)
        if not os.path.exists(log_file):
            pass 

        if os.path.exists(log_file):
            try:
                # Saltamos la primera fila (metadata JSON)
                df = pd.read_csv(log_file, skiprows=1)
                if not df.empty and 'r' in df.columns and 'l' in df.columns:
                    all_rewards.append(df['r'].values)
                    all_lengths.append(df['l'].values)
                    min_len = min(min_len, len(df))
                else:
                    print(f"     Advertencia: Archivo inválido {log_file}")
            except Exception as e:
                print(f"     Error cargando {log_file}: {e}")

    if not all_rewards:
        return None, None, 0

    try:
        if min_len == float('inf'):
            return None, None, 0

        rewards_truncated = [r[:min_len] for r in all_rewards]
        lengths_truncated = [l[:min_len] for l in all_lengths]
        
        return np.array(rewards_truncated), np.array(lengths_truncated), min_len
    except Exception as e:
        print(f"Error al procesar datos: {e}")
        return None, None, 0

def plot_comparison(env_name_base, num_runs, smooth_window, tipo_variant=None):
    """
    Genera los gráficos.
    :param env_name_base: El nombre base del entorno (ej. TwoTigersEnv)
    :param tipo_variant: 'corto', 'largo' o None. Se agrega al nombre del entorno.
    """
    
    # Construir el nombre real del directorio a buscar
    if tipo_variant:
        env_name_search = f"{env_name_base}-{tipo_variant}"
    else:
        env_name_search = env_name_base

    print(f"Iniciando comparativa para: {env_name_search}")
    print("="*60)
    
    algos = {
        'IS-DQN': 'IS-dqn_logs',
        'DQN (Baseline)': 'baselines/dqn_logs',
        'DRQN (Baseline)': 'baselines/drqn_logs',
        'PPO-LSTM (Baseline)': 'baselines/ppo_logs'
    }
    
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    plotted_any = False

    for algo_name, base_path in algos.items():
        print(f"Procesando {algo_name}...")
        rewards, lengths, min_ep_len = load_log_data(base_path, env_name_search, num_runs)
        
        if rewards is None or min_ep_len == 0:
            print(f"  [!] No hay datos válidos para {algo_name} en {env_name_search}.\n")
            continue
        
        print(f"  [OK] {rewards.shape[0]} corridas encontradas. Episodios: {min_ep_len}\n")
        plotted_any = True

        # Suavizar
        rewards_smooth = smooth_data(rewards, smooth_window)
        lengths_smooth = smooth_data(lengths, smooth_window)
        
        # Calcular estadísticas
        mean_r = np.mean(rewards_smooth, axis=0)
        std_r = np.std(rewards_smooth, axis=0)
        mean_l = np.mean(lengths_smooth, axis=0)
        std_l = np.std(lengths_smooth, axis=0)
        
        x_axis = np.arange(min_ep_len)
        
        # Plot Recompensas
        ax1.plot(x_axis, mean_r, label=algo_name, linewidth=2)
        ax1.fill_between(x_axis, mean_r - std_r, mean_r + std_r, alpha=0.2)
        
        # Plot Largos
        ax2.plot(x_axis, mean_l, label=algo_name, linewidth=2)
        ax2.fill_between(x_axis, mean_l - std_l, mean_l + std_l, alpha=0.2)

    if not plotted_any:
        print("\nERROR: No se encontró información de ningún algoritmo para generar la gráfica.")
        print(f"Asegúrate de que existan carpetas como: IS-dqn_logs/{env_name_search}")
        return

    # Configuración de Gráficos
    titulo_suffix = f" ({env_name_search})"
    
    ax1.set_title(f'Recompensa Promedio{titulo_suffix}', fontsize=16)
    ax1.set_xlabel('Episodios', fontsize=12)
    ax1.set_ylabel('Recompensa', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True)
    
    ax2.set_title(f'Largo Promedio{titulo_suffix}', fontsize=16)
    ax2.set_xlabel('Episodios', fontsize=12)
    ax2.set_ylabel('Pasos', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True)
    
    plt.tight_layout()
    output_file = f'comparativa_{env_name_search}.png'
    plt.savefig(output_file)
    print(f"¡Gráfico guardado exitosamente en: {output_file}!")
    # plt.show() 

def main():
    parser = argparse.ArgumentParser(description="Genera gráficos comparativos.")
    
    parser.add_argument('--env', type=str, required=True, 
                        choices=['TwoTigersEnv', 'KeyDoorMazeEnv', 'PODoorEnv', 'DelayedObsEnv'],
                        help='Nombre base del entorno.')
    
    parser.add_argument('--runs', type=int, default=30, 
                        help='Número máximo de corridas a buscar.')
    
    parser.add_argument('--smooth', type=int, default=20, 
                        help='Ventana de suavizado.')

    # --- NUEVO ARGUMENTO ---
    parser.add_argument('--tipo', type=str, default=None, choices=['corto', 'largo','100'],
                        help='Opcional: Agrega "-corto" o "-largo" al nombre del entorno para buscar logs específicos.')
    
    args = parser.parse_args()
    
    if not os.path.exists('baselines') or not os.path.exists('IS-dqn_logs'):
        print("Error: Ejecuta desde la raíz del proyecto (donde están 'baselines' e 'IS-dqn_logs').")
        return
        
    plot_comparison(args.env, args.runs, args.smooth, args.tipo)

if __name__ == '__main__':
    main()