import gymnasium as gym
from gymnasium import spaces
import numpy as np
import io  # Importar io para construir el string del render

class POKeyDoorEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    # 1. Aceptar max_episode_steps en __init__
    def __init__(self, size=10, max_episode_steps=100, render_mode="ansi"):
        self.size = size 
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps # Guardar el límite
        
        self._agent_location = (0, 0)
        self._key_location = (0, 0)
        self._door_location = (0, 0)
        self._has_key = False
        self._grid = np.zeros((size, size), dtype=np.int8)

        # 2. Añadir un contador de pasos
        self._current_steps = 0

        self.action_space = spaces.Discrete(4) 
        self.observation_space = spaces.Box(low=0, high=4, shape=(3, 3), dtype=np.int8)

    def _get_obs(self):
        # ... (sin cambios) ...
        obs = np.ones((3, 3), dtype=np.int8) * 1 
        x, y = self._agent_location
        for i_obs, i_grid in enumerate(range(x - 1, x + 2)):
            for j_obs, j_grid in enumerate(range(y - 1, y + 2)):
                if 0 <= i_grid < self.size and 0 <= j_grid < self.size:
                    obs[i_obs, j_obs] = self._grid[i_grid, j_grid]
                    if (i_grid, j_grid) == self._key_location:
                        obs[i_obs, j_obs] = 2
                    elif (i_grid, j_grid) == self._door_location:
                        obs[i_obs, j_obs] = 3
                    elif (i_grid, j_grid) == self._agent_location:
                        obs[i_obs, j_obs] = 4
        return obs.astype(np.int8)

    def _get_info(self):
        # 3. Incluir los pasos en la info (buena práctica)
        return {
            "agent_location": self._agent_location,
            "has_key": self._has_key,
            "steps": self._current_steps 
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self._grid = np.zeros((self.size, self.size), dtype=np.int8)
        self._grid[0, :] = 1
        self._grid[-1, :] = 1
        self._grid[:, 0] = 1
        self._grid[:, -1] = 1
        
        self._agent_location = (1, 1)
        self._key_location = (self.size - 2, self.size - 2)
        self._door_location = (self.size - 2, 1)
        self._has_key = False
        
        # 4. Reiniciar el contador de pasos
        self._current_steps = 0 
        
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        # ... (lógica de movimiento y colisión) ...
        _action_to_delta = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        delta = _action_to_delta[action]
        current_x, current_y = self._agent_location
        potential_x = current_x + delta[0]
        potential_y = current_y + delta[1]
        terminated = False
        reward = -0.01 
        if self._grid[potential_x, potential_y] == 1:
            reward -= 0.1
        elif (potential_x, potential_y) == self._door_location and not self._has_key:
            reward -= 0.1
        else:
            self._agent_location = (potential_x, potential_y)
        if self._agent_location == self._key_location:
            self._has_key = True
            self._key_location = (-1, -1)
            reward += 0.1
        elif self._agent_location == self._door_location and self._has_key:
            reward += 1.0
            terminated = True
        
        # 5. Incrementar el contador y comprobar el truncado
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
            
            # 6. (Opcional) Añadir el contador al render
            outfile.write(f"Paso: {self._current_steps} / {self.max_episode_steps}\n")
            outfile.write(f"Tiene Llave: {self._has_key}\n")
            
            for row in display_grid:
                outfile.write("".join([char_map.get(cell, "?") for cell in row]) + "\n")
            
            return outfile.getvalue()
        
        else:
            pass