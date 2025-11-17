import csv
import time
import gymnasium as gym
import numpy as np

# IMPORTA TU ENTORNO
from TwoTigersEnv import TwoTigersEnv   # Ajusta el nombre si tu archivo se llama distinto


# Mapeo de acciones del usuario a acción Discrete(9)
ACTION_MAP = {
    "0": 0,  # (0,0)
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8
}

ACTION_DESC = {
    0: "(Escuchar, Escuchar)",
    1: "(Escuchar, Abrir Izq)",
    2: "(Escuchar, Abrir Der)",
    3: "(Abrir Izq, Escuchar)",
    4: "(Abrir Izq, Abrir Izq)",
    5: "(Abrir Izq, Abrir Der)",
    6: "(Abrir Der, Escuchar)",
    7: "(Abrir Der, Abrir Izq)",
    8: "(Abrir Der, Abrir Der)",
}


def imprimir_menu_acciones():
    print("\n===== ACCIONES DISPONIBLES =====")
    for k in ACTION_MAP:
        print(f"  {k}: {ACTION_DESC[ACTION_MAP[k]]}")
    print("================================")


def jugar_manual(episodios=1, csv_output="recompensas.csv"):

    env = TwoTigersEnv(render_mode="human")
    all_rewards = []

    for ep in range(episodios):
        obs, info = env.reset()
        total_reward = 0
        step = 0

        terminado = False
        truncado = False

        while not (terminado or truncado):

            # MOSTRAR MENÚ SIEMPRE
            imprimir_menu_acciones()

            # PEDIR ACCIÓN
            user_input = input("\nElige acción (0-8): ")

            if user_input not in ACTION_MAP:
                print("Acción inválida, intenta nuevamente.")
                continue

            action = ACTION_MAP[user_input]

            # EJECUTAR PASO
            obs, reward, terminado, truncado, info = env.step(action)

            total_reward += reward
            step += 1

            print(f"\nRecompensa paso: {reward}")
            print(f"Recompensa acumulada: {total_reward}")

            # GUARDAR MÉTRICAS
            all_rewards.append({
                "episodio": ep + 1,
                "paso": step,
                "accion": action,
                "accion_desc": ACTION_DESC[action],
                "reward": reward,
                "reward_acumulado": total_reward,
                "obs1": obs[0],
                "obs2": obs[1],
            })

            if truncado:
                print("\n--- Episodio truncado: se alcanzó el límite de pasos ---")

        print(f"\n>>> Episodio {ep+1} terminado. Recompensa total: {total_reward}\n")

    env.close()

    # GUARDAR CSV
    with open(csv_output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_rewards[0].keys())
        writer.writeheader()
        writer.writerows(all_rewards)

    print(f"\nMétricas guardadas en: {csv_output}")


if __name__ == "__main__":
    jugar_manual(episodios=3, csv_output="metricas_recompensas.csv")
