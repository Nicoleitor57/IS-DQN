import gymnasium as gym
import time
import numpy as np
from DelayedObsEnv import DelayedObsEnv
from PODoorEnv import POKeyDoorEnv
from KeyDoorMazeEnv import KeyDoorMazeEnv
from TwoTigersEnv import TwoTigersEnv

Enviroments = {
    "DelayedObsEnv": DelayedObsEnv,
    "POKeyDoorEnv": POKeyDoorEnv,
    "KeyDoorMazeEnv": KeyDoorMazeEnv,
    "TwoTigersEnv": TwoTigersEnv
}


def test_environment(env_name, env_class):
    """
    Función que instancia un entorno y corre un episodio de prueba.
    """
    print(f"\n{'='*40}")
    print(f"🚀 INICIANDO PRUEBA PARA: {env_name}")
    print(f"{'='*40}")
    
    # 1. Instanciar el entorno directamente
    #    Usamos render_mode="ansi" para que imprima texto en la consola.
    try:
        env = env_class(render_mode="ansi")
    except TypeError:
        print(f"ADVERTENCIA: No se pudo pasar 'render_mode'. Intentando sin él.")
        env = env_class()
    except Exception as e:
        print(f"Error al instanciar {env_name}: {e}")
        return

    # 2. Reiniciar el entorno
    try:
        print("Reiniciando entorno...")
        obs, info = env.reset(seed=42)
    except Exception as e:
        print(f"Error en env.reset(): {e}")
        return

    print("\n--- 🌎 ESTADO INICIAL (RENDER) ---")
    print(env.render())
    print("--- 🤖 VISTA INICIAL (OBS) ---")
    print(obs)
    print(f"Info: {info}")

    terminated = False
    truncated = False
    total_reward = 0
    step_count = 0

    # 3. Bucle de prueba (un episodio)
    while not terminated and not truncated:
        step_count += 1
        
        # 4. Elegir una acción aleatoria del espacio de acciones
        action = env.action_space.sample()
        
        # 5. Ejecutar la acción
        try:
            obs, reward, terminated, truncated, info = env.step(action)
        except Exception as e:
            print(f"Error en env.step(): {e}")
            break
            
        total_reward += reward
        
        # 6. Mostrar resultados del paso
        print(f"\n--- PASO {step_count} ---")
        print(f"Acción Aleatoria Tomada: {action}")
        
        print("\n--- 🌎 ESTADO ACTUAL (RENDER) ---")
        print(env.render())
        print("--- 🤖 VISTA DEL AGENTE (OBS) ---")
        print(obs)
        
        print(f"Recompensa: {reward:.2f} (Total: {total_reward:.2f})")
        print(f"Info: {info}")
        print(f"Terminated: {terminated}, Truncated: {truncated}")
        
        # Pausa para poder ver la salida
        time.sleep(0.3) 

    print(f"\n{'='*40}")
    print("🏁 PRUEBA FINALIZADA")
    print(f"Pasos totales: {step_count}")
    print(f"Recompensa final del episodio: {total_reward:.2f}")
    print(f"{'='*40}")
    
    # 7. Cerrar el entorno
    env.close()

# --- Punto de entrada principal del script ---
if __name__ == "__main__":
    
    # VVV --- ¡CAMBIA ESTA LÍNEA PARA PROBAR OTRO ENTORNO! --- VVV
    #env_to_test = "POKeyDoorEnv"
    #env_to_test = "DelayedObsEnv"
    env_to_test = "KeyDoorMazeEnv"
    #env_to_test = "TwoTigersEnv"
    # ^^^ ---------------------------------------------------- ^^^

    if env_to_test in Enviroments:
        chosen_class = Enviroments[env_to_test]
        test_environment(env_to_test, chosen_class)
    else:
        print(f"Error: Entorno '{env_to_test}' no encontrado.")
        print("Asegúrate de que está en el diccionario 'Enviroments' y que el archivo .py está importado.")