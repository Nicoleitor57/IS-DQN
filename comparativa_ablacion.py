# import os
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import argparse
# import sys
# import warnings

# # Suprimir advertencias de Matplotlib/Seaborn por el estilo.
# warnings.filterwarnings("ignore", category=UserWarning)


# def smooth_data(data, window_size):
#     """
#     Suaviza los datos usando una media móvil.
#     """
#     if data.ndim == 1:
#         s = pd.Series(data)
#         return s.rolling(window_size, min_periods=1).mean().values
#     elif data.ndim == 2:
#         smoothed_runs = []
#         for run in data:
#             s = pd.Series(run)
#             smoothed_runs.append(s.rolling(window_size, min_periods=1).mean().values)
#         return np.array(smoothed_runs)
#     return None


# def load_log_data_ablation(algo_path, num_runs):
#     """
#     Carga los datos de monitor.csv para una variante dentro de ./ablacion
#     """
#     all_rewards = []
#     all_lengths = []
#     min_len = float('inf')

#     # print(f"  -> Buscando logs en: {algo_path}")

#     if not os.path.exists(algo_path):
#         # print(f"     Advertencia: No existe {algo_path}")
#         return None, None, 0, 0

#     for i in range(1, num_runs + 1):
#         log_file = os.path.join(algo_path, f"run_{i}", "monitor.csv")

#         if os.path.exists(log_file):
#             try:
#                 # Saltamos la primera línea de metadata
#                 df = pd.read_csv(log_file, skiprows=1) 
#                 if not df.empty and "r" in df.columns and "l" in df.columns:
#                     all_rewards.append(df["r"].values)
#                     all_lengths.append(df["l"].values)
#                     min_len = min(min_len, len(df))
#                 # else:
#                 #     print(f"     Advertencia: Archivo inválido {log_file}")
#             except Exception:
#                 # print(f"     Error cargando {log_file}: {e}")
#                 pass

#     if not all_rewards:
#         return None, None, 0, 0

#     rewards_truncated = [r[:min_len] for r in all_rewards]
#     lengths_truncated = [l[:min_len] for l in all_lengths]

#     return np.array(rewards_truncated), np.array(lengths_truncated), min_len, len(all_rewards)


# def plot_comparison_ablation(env_prefix, num_runs, smooth_window, max_episodes_to_plot=None):
#     """
#     Compara todas las carpetas dentro de ./ablacion que empiecen por env_prefix.
#     """

#     ROOT = "./ablacion"
#     plt.style.use('seaborn-v0_8-darkgrid')

#     print(f"Iniciando comparación para prefijo: {env_prefix}")
#     print("=" * 60)

#     # 1. Descubrir todas las variantes (carpetas que coinciden con el prefijo)
#     variants = [
#         f for f in os.listdir(ROOT)
#         if f.startswith(env_prefix) and os.path.isdir(os.path.join(ROOT, f))
#     ]

#     if not variants:
#         print("ERROR: No se encontraron carpetas de ablación con ese prefijo dentro de ./ablacion/")
#         return

#     # Mapeo de nombres largos a etiquetas cortas para la leyenda
#     name_map = {
#         f"{env_prefix}-PER-H": "Completo (PSR+H+PER)",
#         f"{env_prefix}-H": "Sin PER (PSR+H)",
#         f"{env_prefix}-PER": "Sin Entropía (PSR+PER)",
#         f"{env_prefix}": "PSR Básico (Solo PSR)"
#     }
    
#     # Asegurar orden de graficado: Completo -> Sin PER -> Sin H -> Básico
#     # Construimos un orden basado en las variantes descubiertas.
#     sorted_variants = []
#     for suffix in ["PER-H", "H", "PER", ""]:
#         full_name = f"{env_prefix}-{suffix}" if suffix else env_prefix
#         if full_name in variants:
#             sorted_variants.append(full_name)


#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

#     global_min_ep_len = float('inf')
#     data_dict = {}

#     # 2. Cargar todos los datos para encontrar la longitud mínima común
#     for variant in variants:
#         path = os.path.join(ROOT, variant)
#         rewards, lengths, min_len, count = load_log_data_ablation(path, num_runs)

#         if rewards is None or count == 0:
#             print(f"  [!] Ignorando {variant} (sin datos válidos).")
#             continue

#         data_dict[variant] = {"rewards": rewards, "lengths": lengths}
#         global_min_ep_len = min(global_min_ep_len, min_len)

#     if global_min_ep_len == float('inf'):
#         print("ERROR: No hay corridas válidas para comparación.")
#         return

#     # 3. Aplicar límite de episodios y truncar
#     final_ep_len = global_min_ep_len
#     if max_episodes_to_plot and max_episodes_to_plot > 0:
#         final_ep_len = min(final_ep_len, max_episodes_to_plot)

#     print(f"Truncando todos los logs a {final_ep_len} episodios.")
#     print("=" * 60)

#     # 4. Graficar
#     for variant in sorted_variants:
#         if variant not in data_dict:
#             continue
            
#         data = data_dict[variant]
        
#         rewards = data["rewards"][:, :final_ep_len]
#         lengths = data["lengths"][:, :final_ep_len]

#         # Suavizado
#         rewards_smooth = smooth_data(rewards, smooth_window)

#         # Calcular estadísticas (STD sobre datos no suavizados)
#         mean_r = np.mean(rewards_smooth, axis=0)
#         std_r = np.std(rewards, axis=0)

#         mean_l = np.mean(smooth_data(lengths, smooth_window), axis=0)

#         x = np.arange(final_ep_len)

#         # Etiqueta corta y limpia
#         label = name_map.get(variant, variant)

#         # Plot Recompensas
#         ax1.plot(x, mean_r, label=label, linewidth=2)
#         ax1.fill_between(x, mean_r - std_r, mean_r + std_r, alpha=0.1)

#         # Plot Largos
#         ax2.plot(x, mean_l, label=label, linewidth=2)


#     ax1.set_title(f"Recompensa Promedio (Ablación: {env_prefix})", fontsize=16)
#     ax1.set_xlabel("Episodios", fontsize=12)
#     ax1.set_ylabel("Recompensa", fontsize=12)
#     ax1.legend(fontsize=10)
#     ax1.grid(True)

#     ax2.set_title(f"Largo Promedio (Ablación: {env_prefix})", fontsize=16)
#     ax2.set_xlabel("Episodios", fontsize=12)
#     ax2.set_ylabel("Pasos", fontsize=12)
#     ax2.legend(fontsize=10)
#     ax2.grid(True)

#     plt.tight_layout()
#     output_file = f"ablacion_{env_prefix}.png"
#     plt.savefig(output_file)
#     print(f"\n✔ Imagen guardada en: {output_file}")


# def main():
#     parser = argparse.ArgumentParser(description="Comparación de variantes de ablation dentro de ./ablacion")

#     parser.add_argument("--env", type=str, required=True,
#                         help="Prefijo de carpetas dentro de ./ablacion/, ej: 'Delayed-IS-DQN' o 'IS-DQN'")

#     parser.add_argument("--runs", type=int, default=5, help="Número máximo de corridas a buscar")
#     parser.add_argument("--smooth", type=int, default=20, help="Ventana de suavizado")
#     parser.add_argument("--max-episodes", type=int, default=None,
#                         help="Truncar episodios para visualización")

#     args = parser.parse_args()

#     if not os.path.exists("./ablacion"):
#         print("ERROR: No existe ./ablacion en este directorio. Ejecuta desde la raíz del proyecto.")
#         sys.exit(1)

#     plot_comparison_ablation(args.env, args.runs, args.smooth, args.max_episodes)


# if __name__ == "__main__":
#     main()

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import warnings

# Suprimir advertencias de Matplotlib/Seaborn por el estilo.
warnings.filterwarnings("ignore", category=UserWarning)


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


def load_log_data_ablation(algo_path, num_runs):
    """
    Carga los datos de monitor.csv para una variante.
    """
    all_rewards = []
    all_lengths = []
    min_len = float('inf')

    # print(f"  -> Buscando logs en: {algo_path}")

    if not os.path.exists(algo_path):
        # print(f"     Advertencia: No existe {algo_path}")
        return None, None, 0, 0

    for i in range(1, num_runs + 1):
        log_file = os.path.join(algo_path, f"run_{i}", "monitor.csv")

        if os.path.exists(log_file):
            try:
                # Saltamos la primera línea de metadata
                df = pd.read_csv(log_file, skiprows=1) 
                if not df.empty and "r" in df.columns and "l" in df.columns:
                    all_rewards.append(df["r"].values)
                    all_lengths.append(df["l"].values)
                    min_len = min(min_len, len(df))
                # else:
                #     print(f"     Advertencia: Archivo inválido {log_file}")
            except Exception:
                # print(f"     Error cargando {log_file}: {e}")
                pass

    if not all_rewards:
        return None, None, 0, 0

    rewards_truncated = [r[:min_len] for r in all_rewards]
    lengths_truncated = [l[:min_len] for l in all_lengths]

    return np.array(rewards_truncated), np.array(lengths_truncated), min_len, len(all_rewards)


# 💡 CAMBIO: Se añadió 'experiment_name' como primer argumento.
def plot_comparison_ablation(experiment_name, env_prefix, num_runs, smooth_window, max_episodes_to_plot=None):
    """
    Compara todas las carpetas dentro de ./ablacion/<experiment_name> que empiecen por env_prefix.
    """

    # 💡 CAMBIO CLAVE: La ruta raíz ahora incluye el nombre del experimento
    ROOT = os.path.join("./ablacion", experiment_name) 
    plt.style.use('seaborn-v0_8-darkgrid')

    print(f"Iniciando comparación para experimento: {experiment_name}")
    print(f"Buscando variantes con prefijo '{env_prefix}' dentro de: {ROOT}")
    print("=" * 60)

    # 1. Descubrir todas las variantes (carpetas que coinciden con el prefijo)
    # Asegúrate de que ROOT exista, aunque ya se chequeó en main()
    if not os.path.exists(ROOT):
        print(f"ERROR: El directorio base {ROOT} no existe.")
        return

    variants = [
        f for f in os.listdir(ROOT)
        if f.startswith(env_prefix) and os.path.isdir(os.path.join(ROOT, f))
    ]

    if not variants:
        print(f"ERROR: No se encontraron carpetas con prefijo '{env_prefix}' dentro de {ROOT}")
        return

    # Mapeo de nombres largos a etiquetas cortas para la leyenda
    name_map = {
        f"{env_prefix}-PER-H": "Completo (PSR+H+PER)",
        f"{env_prefix}-H": "Sin PER (PSR+H)",
        f"{env_prefix}-PER": "Sin Entropía (PSR+PER)",
        f"{env_prefix}": "PSR Básico (Solo PSR)"
    }
    
    # Asegurar orden de graficado: Completo -> Sin PER -> Sin H -> Básico
    # Construimos un orden basado en las variantes descubiertas.
    sorted_variants = []
    for suffix in ["PER-H", "H", "PER", ""]:
        full_name = f"{env_prefix}-{suffix}" if suffix else env_prefix
        if full_name in variants:
            sorted_variants.append(full_name)


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    global_min_ep_len = float('inf')
    data_dict = {}

    # 2. Cargar todos los datos para encontrar la longitud mínima común
    for variant in sorted_variants: # Usar sorted_variants para el orden de carga también
        path = os.path.join(ROOT, variant)
        rewards, lengths, min_len, count = load_log_data_ablation(path, num_runs)

        if rewards is None or count == 0:
            print(f"  [!] Ignorando {variant} (sin datos válidos).")
            continue

        data_dict[variant] = {"rewards": rewards, "lengths": lengths}
        global_min_ep_len = min(global_min_ep_len, min_len)

    if global_min_ep_len == float('inf'):
        print("ERROR: No hay corridas válidas para comparación.")
        return

    # 3. Aplicar límite de episodios y truncar
    final_ep_len = global_min_ep_len
    if max_episodes_to_plot and max_episodes_to_plot > 0:
        final_ep_len = min(final_ep_len, max_episodes_to_plot)

    print(f"Truncando todos los logs a {final_ep_len} episodios.")
    print("=" * 60)

    # 4. Graficar
    for variant in sorted_variants:
        if variant not in data_dict:
            continue
            
        data = data_dict[variant]
        
        # Truncar los datos
        rewards = data["rewards"][:, :final_ep_len]
        lengths = data["lengths"][:, :final_ep_len]

        # Suavizado
        rewards_smooth = smooth_data(rewards, smooth_window)

        # Calcular estadísticas (STD sobre datos no suavizados)
        mean_r = np.mean(rewards_smooth, axis=0)
        std_r = np.std(rewards, axis=0) # STD sobre los datos sin suavizar

        mean_l = np.mean(smooth_data(lengths, smooth_window), axis=0)

        x = np.arange(final_ep_len)

        # Etiqueta corta y limpia
        label = name_map.get(variant, variant)

        # Plot Recompensas
        ax1.plot(x, mean_r, label=label, linewidth=2)
        ax1.fill_between(x, mean_r - std_r, mean_r + std_r, alpha=0.1)

        # Plot Largos
        ax2.plot(x, mean_l, label=label, linewidth=2)


    ax1.set_title(f"Recompensa Promedio (Ablación: {experiment_name} / {env_prefix})", fontsize=16)
    ax1.set_xlabel("Episodios", fontsize=12)
    ax1.set_ylabel("Recompensa", fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True)

    ax2.set_title(f"Largo Promedio (Ablación: {experiment_name} / {env_prefix})", fontsize=16)
    ax2.set_xlabel("Episodios", fontsize=12)
    ax2.set_ylabel("Pasos", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True)

    plt.tight_layout()
    # 💡 CAMBIO: Nombre del archivo de salida incluye el experimento para evitar conflictos
    output_file = f"{experiment_name}_{env_prefix}.png" 
    plt.savefig(output_file)
    print(f"\n✔ Imagen guardada en: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Comparación de variantes de ablation dentro de ./ablacion/<EXPERIMENTO>")

    # 💡 CAMBIO: Nuevo argumento para el nombre del experimento
    parser.add_argument("--exp", type=str, required=True,
                        choices=["ablacion_tigers", "ablacion_maze", "ablacion_delayed"],
                        help="Nombre del experimento (subcarpeta dentro de ./ablacion/)")

    parser.add_argument("--env", type=str, required=True,
                        help="Prefijo de carpetas DENTRO de la carpeta del experimento, ej: 'Delayed-IS-DQN' o 'IS-DQN'")

    parser.add_argument("--runs", type=int, default=5, help="Número máximo de corridas a buscar")
    parser.add_argument("--smooth", type=int, default=20, help="Ventana de suavizado")
    parser.add_argument("--max-episodes", type=int, default=None,
                        help="Truncar episodios para visualización")

    args = parser.parse_args()

    # 💡 CAMBIO: Construir la ruta de la carpeta del experimento
    ROOT_DIR = os.path.join("./ablacion", args.exp)

    if not os.path.exists(ROOT_DIR):
        print(f"ERROR: No existe la carpeta del experimento: {ROOT_DIR}. Ejecuta desde la raíz del proyecto.")
        sys.exit(1)

    # 💡 CAMBIO: Pasar el argumento 'exp' a la función de graficado
    plot_comparison_ablation(args.exp, args.env, args.runs, args.smooth, args.max_episodes)


if __name__ == "__main__":
    main()