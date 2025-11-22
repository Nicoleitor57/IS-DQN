import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys

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

    if not os.path.exists(target_dir):
        print(f"     Advertencia: No existe la carpeta {target_dir}")
        return None, None, 0, 0

    for i in range(1, num_runs + 1):
        log_file = os.path.join(target_dir, f'run_{i}', 'monitor.csv')
        
        if os.path.exists(log_file):
            try:
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
        return None, None, 0, 0

    # Truncamos los arrays a la longitud mínima para ese algoritmo
    rewards_truncated = [r[:min_len] for r in all_rewards]
    lengths_truncated = [l[:min_len] for l in all_lengths]
    
    return np.array(rewards_truncated), np.array(lengths_truncated), min_len, len(all_rewards)

# Agregamos 'max_episodes_to_plot' a la firma
def plot_comparison(env_name_base, num_runs, smooth_window, tipo_variant=None, max_episodes_to_plot=None):
    """
    Genera los gráficos.
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
    
    # 1. Determinar la longitud final de los datos (la más corta entre todos los logs y el parámetro)
    global_min_ep_len = float('inf')
    algo_data = {}

    for algo_name, base_path in algos.items():
        # rewards y lengths ya están truncados a la longitud mínima de ESE algoritmo
        rewards, lengths, min_ep_len_algo, valid_runs = load_log_data(base_path, env_name_search, num_runs)
        
        if rewards is None or valid_runs < 1:
            continue
            
        global_min_ep_len = min(global_min_ep_len, min_ep_len_algo)
        algo_data[algo_name] = {'rewards': rewards, 'lengths': lengths}

    if global_min_ep_len == float('inf'):
        print("\nERROR: No se encontró información válida de ningún algoritmo para generar la gráfica.")
        return

    # Aplicar la restricción del usuario
    final_ep_len = global_min_ep_len
    if max_episodes_to_plot is not None and max_episodes_to_plot > 0:
        final_ep_len = min(global_min_ep_len, max_episodes_to_plot)

    if final_ep_len == 0:
        print("\nAdvertencia: El límite de episodios es cero o no hay datos disponibles.")
        return

    print(f"Truncando todos los logs a un máximo de {final_ep_len} episodios.")
    print("="*60)
    
    # 2. Generar Plots
    for algo_name, data in algo_data.items():
        plotted_any = True

        rewards_all_runs = data['rewards']
        lengths_all_runs = data['lengths']

        # Truncamos los arrays a la longitud final común
        rewards_final = rewards_all_runs[:, :final_ep_len]
        lengths_final = lengths_all_runs[:, :final_ep_len]

        print(f"Procesando {algo_name} con {rewards_final.shape[0]} corridas.")

        # Suavizar
        rewards_smooth = smooth_data(rewards_final, smooth_window)
        lengths_smooth = smooth_data(lengths_final, smooth_window)
        
        # Calcular estadísticas
        mean_r = np.mean(rewards_smooth, axis=0)
        std_r = np.std(rewards_final, axis=0) # Usamos STD de los datos sin suavizar para la banda
        mean_l = np.mean(lengths_smooth, axis=0)
        
        x_axis = np.arange(final_ep_len)
        
        # Plot Recompensas
        ax1.plot(x_axis, mean_r, label=algo_name, linewidth=2)
        ax1.fill_between(x_axis, mean_r - std_r, mean_r + std_r, alpha=0.2)
        
        # Plot Largos
        ax2.plot(x_axis, mean_l, label=algo_name, linewidth=2)

    if not plotted_any:
        print("\nERROR: No se encontró información de ningún algoritmo para generar la gráfica.")
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
    print(f"\n¡Gráfico guardado exitosamente en: {output_file}!")
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
    
    parser.add_argument('--tipo', type=str, default=None, choices=['corto', 'largo','100', '1'],
                        help='Opcional: Agrega sufijo al nombre del entorno para buscar logs específicos.')

    # --- NUEVO ARGUMENTO CLAVE ---
    parser.add_argument('--max-episodes', type=int, default=None,
                        help='Máximo número de episodios a incluir en la gráfica (trunca los logs a este valor).')
    # ------------------------------
    
    args = parser.parse_args()
    
    if not os.path.exists('baselines') or not os.path.exists('IS-dqn_logs'):
        print("Error: Ejecuta desde la raíz del proyecto (donde están 'baselines' e 'IS-dqn_logs').")
        sys.exit(1)
        
    # Pasamos el nuevo argumento a la función
    plot_comparison(args.env, args.runs, args.smooth, args.tipo, args.max_episodes)

if __name__ == '__main__':
    main()