import gymnasium as gym
from gymnasium import spaces
import numpy as np
import io  # Importar io para construir el string del render

class StochasticGridEnv(gym.Env):
    """
    Entorno de Laberinto Estocástico (Stochastic Grid)
    
    MODIFICACIONES CLAVE:
    1. Movimiento Estocástico: slip_prob de fallo.
    2. Inicio Aleatorio: El agente comienza en cualquier posición de "Piso".
    """
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, size=10, max_episode_steps=100, render_mode="ansi", slip_prob=0.1):
        self.size = size 
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.slip_prob = slip_prob # <-- NUEVO: Probabilidad de resbalar/fallar
        
        self._agent_location = (0, 0)
        self._key_location = (0, 0)
        self._door_location = (0, 0)
        self._has_key = False
        self._grid = np.zeros((size, size), dtype=np.int8)

        self._current_steps = 0

        self.action_space = spaces.Discrete(4) 
        # La observación (3x3) se mantiene
        self.observation_space = spaces.Box(low=0, high=4, shape=(3, 3), dtype=np.int8)
        
        # Necesario para el sampling estocástico
        self.np_random, seed = gym.utils.seeding.np_random()


    # def _get_obs(self):
        
    #     obs = np.ones((3, 3), dtype=np.int8) * 1 
    #     x, y = self._agent_location
    #     for i_obs, i_grid in enumerate(range(x - 1, x + 2)):
    #         for j_obs, j_grid in enumerate(range(y - 1, y + 2)):
    #             if 0 <= i_grid < self.size and 0 <= j_grid < self.size:
    #                 obs[i_obs, j_obs] = self._grid[i_grid, j_grid]
    #                 if (i_grid, j_grid) == self._key_location:
    #                     obs[i_obs, j_grid] = 2
    #                 elif (i_grid, j_grid) == self._door_location:
    #                     obs[i_obs, j_grid] = 3
    #                 elif (i_grid, j_grid) == self._agent_location:
    #                     obs[i_obs, j_grid] = 4
    #     return obs.astype(np.int8)
    def _get_obs(self):
        # Inicializamos la observación local a 1 (Muro, por si está fuera de límites)
        obs = np.ones((3, 3), dtype=np.int8) * 1 
        x, y = self._agent_location
        
        # Recorremos la ventana de 3x3
        for i_obs, i_grid in enumerate(range(x - 1, x + 2)):
            for j_obs, j_grid in enumerate(range(y - 1, y + 2)):
                
                # Verificamos si la coordenada global está dentro de la cuadrícula
                if 0 <= i_grid < self.size and 0 <= j_grid < self.size:
                    
                    # --- CÓDIGO CORREGIDO: Usamos i_obs, j_obs para acceder a 'obs' ---
                    
                    # 1. Base Grid (Piso/Muro)
                    obs[i_obs, j_obs] = self._grid[i_grid, j_grid]
                    
                    # 2. Llave (K)
                    if (i_grid, j_grid) == self._key_location:
                        obs[i_obs, j_obs] = 2 # K
                        
                    # 3. Puerta (D)
                    elif (i_grid, j_grid) == self._door_location:
                        obs[i_obs, j_obs] = 3 # D
                        
                    # 4. Agente (A) - Siempre en (1, 1) local
                    elif (i_grid, j_grid) == self._agent_location:
                        obs[i_obs, j_obs] = 4 # A
                        
        return obs.astype(np.int8)
    
    def _get_info(self):
        return {
            "agent_location": self._agent_location,
            "has_key": self._has_key,
            "steps": self._current_steps 
        }

    def reset(self, seed=None, options=None):
        # 1. Resetear aleatoriedad
        super().reset(seed=seed)
        
        # 2. Reconstruir el grid (paredes)
        self._grid = np.zeros((self.size, self.size), dtype=np.int8)
        self._grid[0, :] = 1
        self._grid[-1, :] = 1
        self._grid[:, 0] = 1
        self._grid[:, -1] = 1
        
        # 3. INICIO ALEATORIO <-- NUEVO
        valid_starts = np.argwhere(self._grid == 0) # Encuentra todas las coordenadas de Piso
        if len(valid_starts) > 0:
            idx = self.np_random.integers(0, len(valid_starts))
            self._agent_location = tuple(valid_starts[idx])
        else:
            self._agent_location = (1, 1) # Fallback
        
        # 4. Posiciones de objetos
        self._key_location = (self.size - 2, self.size - 2)
        self._door_location = (self.size - 2, 1)
        self._has_key = False
        
        # 5. Resetear contador
        self._current_steps = 0 
        
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        _action_to_delta = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        delta = _action_to_delta[action]
        current_x, current_y = self._agent_location
        
        # --- ESTOCASTICIDAD EN LA TRANSICIÓN (T) <-- NUEVO ---
        if self.np_random.random() < self.slip_prob:
            # Movimiento falla: se queda en el mismo sitio (delta = 0, 0)
            actual_delta_x, actual_delta_y = 0, 0
        else:
            # Movimiento exitoso
            actual_delta_x, actual_delta_y = delta[0], delta[1]
            
        potential_x = current_x + actual_delta_x
        potential_y = current_y + actual_delta_y
        # -----------------------------------------------------
        
        terminated = False
        reward = -0.01 

        # 1. Chequeo de colisión (Muro o Puerta sin llave)
        new_location_is_wall = self._grid[potential_x, potential_y] == 1
        new_location_is_locked_door = (potential_x, potential_y) == self._door_location and not self._has_key
        
        if new_location_is_wall or new_location_is_locked_door:
            # Recibe penalización y se queda en la ubicación actual.
            reward -= 0.1
        else:
            # Movimiento válido: actualiza la ubicación.
            self._agent_location = (potential_x, potential_y)
        
        # 2. Lógica de interacción (Recoger llave / Abrir puerta)
        if self._agent_location == self._key_location:
            self._has_key = True
            self._key_location = (-1, -1)
            reward += 0.1
        elif self._agent_location == self._door_location and self._has_key:
            reward += 1.0
            terminated = True
        
        # 3. Fin del paso
        self._current_steps += 1
        truncated = self._current_steps >= self.max_episode_steps

        observation = self._get_obs()
        info = self._get_info() 

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "ansi":
            char_map = {0: " ", 1: "█", 2: "K", 3: "D", 4: "A"}
            display_grid = self._grid.copy()
            
            if self._key_location != (-1, -1):
                key_x, key_y = self._key_location
                display_grid[key_x, key_y] = 2
            
            door_x, door_y = self._door_location
            display_grid[door_x, door_y] = 3
            
            agent_x, agent_y = self._agent_location
            display_grid[agent_x, agent_y] = 4
            
            outfile = io.StringIO()
            
            outfile.write(f"Paso: {self._current_steps} / {self.max_episode_steps}\n")
            outfile.write(f"Llave: {self._has_key}\n")
            
            for row in display_grid:
                outfile.write("".join([char_map.get(cell, "?") for cell in row]) + "\n")
            
            return outfile.getvalue()
        
        else:
            pass

    def close(self):
        pass