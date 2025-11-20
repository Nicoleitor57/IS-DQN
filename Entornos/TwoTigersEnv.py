import gymnasium as gym
from gymnasium import spaces
import numpy as np
import io
import os   
import time

class TwoTigersEnv(gym.Env):
    """
    Entorno de los Dos Tigres Factoreados (Compatible con Gymnasium)
    
    Objetivo:
    El agente se enfrenta a dos problemas de POMDP "Tigre" independientes 
    y simultáneos. Cada problema tiene dos puertas (Izquierda, Derecha).
    
    Problema 1: Tigre/Tesoro detrás de Puerta 1-L o 1-R.
    Problema 2: Tigre/Tesoro detrás de Puerta 2-L o 2-R.
    
    Estado Latente (Oculto):
    El estado real es S = (S1, S2), donde:
    - S1: Posición del Tigre 1 (0=Izquierda, 1=Derecha)
    - S2: Posición del Tigre 2 (0=Izquierda, 1=Derecha)
    S1 y S2 son totalmente independientes.
    
    Acciones (MultiDiscrete[3, 3]):
    El agente debe elegir una acción para AMBOS problemas en cada paso.
    La acción es un vector [accion_1, accion_2], donde cada acción puede ser:
    - 0: Escuchar
    - 1: Abrir Puerta Izquierda
    - 2: Abrir Puerta Derecha
    
    Observaciones (MultiDiscrete[3, 3]):
    La observación es un vector [sonido_1, sonido_2].
    - 0: Sin sonido (estado inicial o si se abrió una puerta)
    - 1: Se oye un tigre a la Izquierda
    - 2: Se oye un tigre a la Derecha
    
    La acción de "Escuchar" (0) da una observación ruidosa.
    - Con `accuracy` (ej. 0.85), el sonido es correcto.
    - Con `1 - accuracy` (ej. 0.15), el sonido es incorrecto.
    
    Recompensas:
    - Abrir puerta con Tesoro: +10
    - Abrir puerta con Tigre: -100
    - Escuchar: -1
    Las recompensas de ambas acciones se suman.
    
    Reseteo:
    Cuando se abre una puerta (sea Tigre o Tesoro), el estado de ESE 
    problema específico se reinicia aleatoriamente. El otro continúa.
    Esto lo convierte en una tarea continua (manejada con `truncated`).
    """

    metadata = {"render_modes": ["ansi", "human"]}

    def __init__(self, accuracy=0.85, max_episode_steps=200, render_mode="ansi"):
        super().__init__()
        
        self.accuracy = accuracy
        self.max_episode_steps = max_episode_steps
        self._render_mode = render_mode # El usuario selecciona el modo aquí
        
        self.R_TREASURE = 10
        self.R_TIGER = -100
        self.R_LISTEN = -1

        #self.action_space = spaces.MultiDiscrete([3, 3])
        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.MultiDiscrete([3, 3])

        # Estado interno
        self._tiger_pos_1 = 0
        self._tiger_pos_2 = 0
        self._current_steps = 0
        
        # --- Nuevos atributos para renderizado humano ---
        self._last_action = None
        self._last_observation = None
        self._last_reward = 0
        
    def action_space_(self):
        return self.action_space.n
    
    def state_space_(self):
        return 4 #( S_rr, S_rl, S_lr, S_ll )

    def _get_info(self):
        return {
            "tiger_1_pos": "Left" if self._tiger_pos_1 == 0 else "Right",
            "tiger_2_pos": "Left" if self._tiger_pos_2 == 0 else "Right",
            "steps": self._current_steps
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self._tiger_pos_1 = self.np_random.integers(0, 2)
        self._tiger_pos_2 = self.np_random.integers(0, 2)
        self._current_steps = 0
        
        observation = np.array([0, 0], dtype=np.int8)
        info = self._get_info()
        
        # --- Actualización para renderizado ---
        self._last_action = None
        self._last_observation = observation
        self._last_reward = 0
        
        if self._render_mode == "human":
            self.render()
        # -------------------------------------
        
        return observation, info

    def step(self, action):
        # ... (La lógica interna de 'step' es idéntica) ...
        #act_1, act_2 = action
        act_1 = action // 3  # División entera (0-2)
        act_2 = action % 3   # Módulo (0-2)
        
        # Guardamos la acción combinada para el render
        self._last_action_combined = action
        
        total_reward = 0
        obs_1 = 0
        obs_2 = 0

        # --- Lógica del Problema 1 ---
        if act_1 == 0:
            total_reward += self.R_LISTEN
            obs_1 = (1 if self._tiger_pos_1 == 0 else 2) if self.np_random.random() < self.accuracy else (2 if self._tiger_pos_1 == 0 else 1)
        elif act_1 == 1:
            total_reward += self.R_TIGER if self._tiger_pos_1 == 0 else self.R_TREASURE
            self._tiger_pos_1 = self.np_random.integers(0, 2)
        elif act_1 == 2:
            total_reward += self.R_TIGER if self._tiger_pos_1 == 1 else self.R_TREASURE
            self._tiger_pos_1 = self.np_random.integers(0, 2)

        # --- Lógica del Problema 2 ---
        if act_2 == 0:
            total_reward += self.R_LISTEN
            obs_2 = (1 if self._tiger_pos_2 == 0 else 2) if self.np_random.random() < self.accuracy else (2 if self._tiger_pos_2 == 0 else 1)
        elif act_2 == 1:
            total_reward += self.R_TIGER if self._tiger_pos_2 == 0 else self.R_TREASURE
            self._tiger_pos_2 = self.np_random.integers(0, 2)
        elif act_2 == 2:
            total_reward += self.R_TIGER if self._tiger_pos_2 == 1 else self.R_TREASURE
            self._tiger_pos_2 = self.np_random.integers(0, 2)

        # --- Finalizar el paso ---
        self._current_steps += 1
        terminated = False 
        truncated = self._current_steps >= self.max_episode_steps
        
        observation = np.array([obs_1, obs_2], dtype=np.int8)
        info = self._get_info()

        # --- Actualización para renderizado ---
        self._last_action = action
        self._last_observation = observation
        self._last_reward = total_reward
        
        if self._render_mode == "human":
            self.render()
        # -------------------------------------

        return observation, total_reward, terminated, truncated, info

    def render(self):
        
        # --- MODO ANSI (devuelve un string) ---
        if self._render_mode == "ansi":
            outfile = io.StringIO()
            outfile.write(f"Paso: {self._current_steps}\n")
            outfile.write(f"Estado Real (Oculto):\n")
            outfile.write(f"  Tigre 1: {'Izquierda' if self._tiger_pos_1 == 0 else 'Derecha'}\n")
            outfile.write(f"  Tigre 2: {'Izquierda' if self._tiger_pos_2 == 0 else 'Derecha'}\n")
            return outfile.getvalue()
        
        # --- MODO HUMAN (imprime un dashboard en la consola) ---
        elif self._render_mode == "human":
            # Limpiar la consola
            os.system('cls' if os.name == 'nt' else 'clear')
            
            print(f"====== DOS TIGRES (Paso: {self._current_steps}) ======")
            print(f"Precisión del Sonido: {self.accuracy * 100}%\n")

            # --- Problema 1 ---
            print("--- Problema 1 ---")
            pos1 = self._tiger_pos_1
            # [T, G] o [G, T]
            doors1_visual = "[ 🐯 Tigre | 💰 Tesoro ]" if pos1 == 0 else "[ 💰 Tesoro | 🐯 Tigre ]"
            print(f"Estado Real: {doors1_visual}")
            
            # --- Problema 2 ---
            print("\n--- Problema 2 ---")
            pos2 = self._tiger_pos_2
            doors2_visual = "[ 🐯 Tigre | 💰 Tesoro ]" if pos2 == 0 else "[ 💰 Tesoro | 🐯 Tigre ]"
            print(f"Estado Real: {doors2_visual}")
            
            # --- Información del Agente ---
            print("\n" + "="*30)
            print("Información del Agente:")
            
            # Última Acción
            # if self._last_action is not None:
            #     act1_str = ["Escuchar", "Abrir Izq", "Abrir Der"][self._last_action[0]]
            #     act2_str = ["Escuchar", "Abrir Izq", "Abrir Der"][self._last_action[1]]
            #     print(f"Última Acción:  [{act1_str}, {act2_str}]")
            # Última Acción
            if self._last_action is not None:
                # Decodifica la acción (un int 0-8) igual que en step()
                act_1 = self._last_action // 3 
                act_2 = self._last_action % 3
                
                act1_str = ["Escuchar", "Abrir Izq", "Abrir Der"][act_1]
                act2_str = ["Escuchar", "Abrir Izq", "Abrir Der"][act_2]
                print(f"Última Acción:  [{act1_str}, {act2_str}]")
            else:
                print("Última Acción:  [N/A (Inicio de episodio)]")

            # Observación (lo que el agente "oye")
            if self._last_observation is not None:
                obs1_str = ["Silencio", "Sonido Izq", "Sonido Der"][self._last_observation[0]]
                obs2_str = ["Silencio", "Sonido Izq", "Sonido Der"][self._last_observation[1]]
                print(f"Observación:    [{obs1_str}, {obs2_str}]")
            else:
                print("Observación:    [N/A]")
            
            print(f"Recompensa:     {self._last_reward}")
            print("\n")
            
            # Pausa breve para que el humano pueda leer
            time.sleep(0.2)

    def close(self):
        pass