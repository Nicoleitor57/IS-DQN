import gymnasium as gym
from gymnasium import spaces
import numpy as np
import io

class KeyDoorMazeEnv(gym.Env):
    """
    Entorno de Laberinto con Llave y Puerta (PO-KeyDoor) - Diseño "Encrucijada"
    
    Compatible con Gymnasium.
    
    Objetivo:
    El agente debe recoger una llave (Roja o Azul) y luego ir a la
    puerta del color correspondiente para obtener una gran recompensa.
    Elegir la puerta incorrecta resulta en un gran castigo.
    
    Observabilidad Parcial:
    El agente solo ve una cuadrícula de 3x3 a su alrededor.
    
    Valores del Grid:
    - 0: Piso (vacío)
    - 1: Muro
    - 2: Llave Roja
    - 3: Llave Azul
    - 4: Puerta Roja
    - 5: Puerta Azul
    
    
    Espacio de Observación (3x3 Box):
    Igual que los valores del grid, pero 6 es el Agente (en la celda 1,1).
    
    Acciones:
    - 0: Arriba
    - 1: Abajo
    - 2: Izquierda
    - 3: Derecha
    
    Estado Info-Estructural (Oculto):
    El agente debe recordar qué llave cogió (Roja o Azul) al inicio de
    un pasillo largo y ambiguo para saber qué puerta elegir al final.
    """
    def __init__(self, height=15, width=19, max_episode_steps=250, render_mode="ansi"):
        # Dimensiones (impares) para un centrado adecuado
        assert height >= 7 and height % 2 == 1, "Altura debe ser impar >= 7"
        assert width >= 11 and width % 2 == 1, "Anchura debe ser impar >= 11"
        
        self.height = height
        self.width = width
        self.max_episode_steps = max_episode_steps
        self._render_mode = render_mode
        self.window_size = 512

        # --- Generar el Mapa "Serpiente Doble" ---
        self._grid = np.zeros((height, width), dtype=np.int8) # Empezar con piso
        
        # 1. Paredes exteriores
        self._grid[0, :] = 1
        self._grid[-1, :] = 1
        self._grid[:, 0] = 1
        self._grid[:, -1] = 1
        
        # 2. Muro divisor central
        mid_col = width // 2
        self._grid[1:-1, mid_col] = 1

        # 3. Tallo inicial y punto de decisión (Encrucijada)
        start_row = height - 2
        self._start_pos = (start_row, mid_col)
        self._grid[start_row - 1 : start_row + 1, mid_col] = 0 # Tallo
        
        fork_row = start_row - 2
        self._grid[fork_row, 1:mid_col] = 0   # Conexión izquierda
        self._grid[fork_row, mid_col+1:-1] = 0 # Conexión derecha

        # 4. Generar las "serpientes" de muros internos con zigzag navegable
        # Laberinto Izquierdo - crear zigzag dejando pasillos abiertos
        for r in range(4, fork_row, 2):  # Cada 2 filas
            if ((r - 4) // 2) % 2 == 0:  # Muro desde la izquierda, pasillo a la derecha
                self._grid[r, 1:mid_col-3] = 1
                # Abrir pasillo vertical
                self._grid[r, mid_col-3] = 0
                self._grid[r+1, mid_col-3] = 0
            else:  # Muro desde la derecha, pasillo a la izquierda
                self._grid[r, 3:mid_col] = 1
                # Abrir pasillo vertical
                self._grid[r, 2] = 0
                self._grid[r+1, 2] = 0
        
        # Laberinto Derecho - crear zigzag dejando pasillos abiertos
        for r in range(4, fork_row, 2):  # Cada 2 filas
            if ((r - 4) // 2) % 2 == 0:  # Muro desde la derecha, pasillo a la izquierda
                self._grid[r, mid_col+3:width-1] = 1
                # Abrir pasillo vertical
                self._grid[r, mid_col+2] = 0
                self._grid[r+1, mid_col+2] = 0
            else:  # Muro desde la izquierda, pasillo a la derecha
                self._grid[r, mid_col+1:width-3] = 1
                # Abrir pasillo vertical
                self._grid[r, width-3] = 0
                self._grid[r+1, width-3] = 0
        
        # 5. Crear los "Vestíbulos" AMPLIADOS (Salas de Decisión)
        # Vestíbulo Izquierdo - Área de 3x6
        for r in range(1, 4):  # Filas 1, 2, 3
            for c in range(2, mid_col - 1):  # Abrir todo el espacio
                self._grid[r, c] = 0
        
        # Vestíbulo Derecho - Área de 3x6
        for r in range(1, 4):  # Filas 1, 2, 3
            for c in range(mid_col + 2, width - 2):  # Abrir todo el espacio
                self._grid[r, c] = 0

        # --- Posiciones de Objetos ---
        self._key_red_pos = (fork_row, 2)
        self._key_blue_pos = (fork_row, width - 3)
        
        # Puertas en los vestíbulos amplios (fila 2 para más espacio de maniobra)
        # Vestíbulo izquierdo
        door_l_pos1 = (2, mid_col - 5)
        door_l_pos2 = (2, mid_col - 3)
        self._door_red_pos = door_l_pos1     # Correcta
        self._door_blue_pos_trap = door_l_pos2  # Trampa

        # Vestíbulo derecho
        door_r_pos1 = (2, mid_col + 3)
        door_r_pos2 = (2, mid_col + 5)
        self._door_blue_pos = door_r_pos1    # Correcta
        self._door_red_pos_trap = door_r_pos2  # Trampa
        
        # --- Espacios de Gymnasium ---
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0, high=6, shape=(3, 3), dtype=np.int8
        )
        
        self._agent_loc = self._start_pos
        self._has_key = 0 # 0=no, 2=roja, 3=azul
        self._episode_steps = 0
        self._current_grid_view = self._grid.copy()

    # def _get_obs(self):
    #     obs = np.ones((3, 3), dtype=np.int8)
    #     x, y = self._agent_loc
    #     for obs_i, grid_i in enumerate(range(x - 1, x + 2)):
    #         for obs_j, grid_j in enumerate(range(y - 1, y + 2)):
    #             if 0 <= grid_i < self.height and 0 <= grid_j < self.width:
    #                 obs[obs_i, obs_j] = self._current_grid_view[grid_i, grid_j]
    #     obs[1, 1] = 6
    #     return obs
    
    def _get_obs(self):
        obs = np.ones((3, 3), dtype=np.int8)
        x, y = self._agent_loc
        for obs_i, grid_i in enumerate(range(x - 1, x + 2)):
            for obs_j, grid_j in enumerate(range(y - 1, y + 2)):
                if 0 <= grid_i < self.height and 0 <= grid_j < self.width:
                    # Simplemente copia el valor del grid
                    obs[obs_i, obs_j] = self._current_grid_view[grid_i, grid_j]
        obs[1, 1] = 6
        return obs

    def _get_info(self):
        return {
            "agent_location": self._agent_loc,
            "has_key": self._has_key,
            "steps": self._episode_steps
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._agent_loc = self._start_pos
        self._has_key = 0
        self._episode_steps = 0
        
        self._current_grid_view = self._grid.copy()
        
        # Colocar Llaves (valor 2 y 3)
        self._current_grid_view[self._key_red_pos] = 2
        self._current_grid_view[self._key_blue_pos] = 3
        
        # Colocar Puertas Correctas (valor 4 y 5)
        self._current_grid_view[self._door_red_pos] = 4
        self._current_grid_view[self._door_blue_pos] = 5
        
        # Colocar Puertas Trampa (valor 7)
        self._current_grid_view[self._door_blue_pos_trap] = 5 
        self._current_grid_view[self._door_red_pos_trap] = 4 

        observation = self._get_obs()
        info = self._get_info()
        
        if self._render_mode == "human":
            self._render_frame()
        return observation, info

    def step(self, action):
        _action_to_delta = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        delta = _action_to_delta[action]
        new_x, new_y = self._agent_loc[0] + delta[0], self._agent_loc[1] + delta[1]
        
        reward = -0.01 
        terminated = False
        
        # Comprobar si la nueva posición es un muro (valor 1)
        if self._current_grid_view[new_x, new_y] == 1: # Choca con muro
            reward -= 0.1 
        else:
            # Movimiento válido
            self._agent_loc = (new_x, new_y)
            # Valor de la celda a la que se movió
            cell_value = self._current_grid_view[self._agent_loc]
            
            # --- Lógica de Interacción ---
            
            if cell_value == 2: # Recoge Llave Roja
                if self._has_key == 0:
                    self._has_key = 2
                    self._current_grid_view[self._key_red_pos] = 0 # Quitar
                    reward += 0.1
            
            elif cell_value == 3: # Recoge Llave Azul
                if self._has_key == 0:
                    self._has_key = 3
                    self._current_grid_view[self._key_blue_pos] = 0 # Quitar
                    reward += 0.1

            # El agente está en una celda con valor 4 (Puerta Roja)
            elif cell_value == 4: 
                if self._has_key == 2: # Y tiene la Llave Roja
                    reward += 10.0 # Éxito
                    terminated = True
                elif self._has_key == 3: # Y tiene la Llave Azul
                    reward -= 5.0 # Fracaso
                    terminated = True
            
            # El agente está en una celda con valor 5 (Puerta Azul)
            elif cell_value == 5: 
                if self._has_key == 3: # Y tiene la Llave Azul
                    reward += 10.0 # Éxito
                    terminated = True
                elif self._has_key == 2: # Y tiene la Llave Roja
                    reward -= 5.0 # Fracaso
                    terminated = True
            
            # # El agente está en una trampa (valor 7)
            # elif cell_value == 7:
            #     if self._has_key != 0:  # Si tiene cualquier llave
            #         reward -= 5.0  # Fracaso (es una trampa)
            #         terminated = True
        
        self._episode_steps += 1
        truncated = self._episode_steps >= self.max_episode_steps
        
        observation = self._get_obs()
        info = self._get_info()

        if self._render_mode == "human":
            self._render_frame()
        return observation, reward, terminated, truncated, info

    # def render(self):
    #     if self._render_mode == "ansi":
    #         # R=LlaveRoja, B=LlaveAzul, D=PuertaRoja, d=PuertaAzul, T=Trampa
    #         char_map = {0: " ", 1: "█", 2: "R", 3: "B", 4: "D", 5: "d", 6: "A"}
            
    #         outfile = io.StringIO()
    #         grid_with_agent = self._current_grid_view.copy()
    #         if 0 <= self._agent_loc[0] < self.height and 0 <= self._agent_loc[1] < self.width:
    #             grid_with_agent[self._agent_loc] = 6 
            
    #         outfile.write(f"Paso: {self._episode_steps} / {self.max_episode_steps}\n")
    #         outfile.write(f"Llave: {self._has_key} (0=No, 2=R, 3=B)\n")
            
    #         for row in grid_with_agent:
    #             outfile.write("".join([char_map[cell] for cell in row]) + "\n")
            
    #         return outfile.getvalue()
    #     elif self._render_mode == "human":
    #         return None
    
    def render(self):
        if self._render_mode == "ansi":
            # Mapa de caracteres base
            char_map = {0: " ", 1: "█", 2: "R", 3: "B", 4: "D", 5: "d", 6: "A"}

            outfile = io.StringIO()

            # Copia el grid para modificarlo SOLO para el render
            grid_para_render = self._current_grid_view.copy()

            # --- HARDCODEO PARA RENDER ---
            # Sobreescribe las celdas de trampa con un número
            # que no usemos, como 9 (solo para el render).
            # (No podemos usar 7 porque lo eliminamos)
            grid_para_render[self._door_blue_pos_trap] = 9
            grid_para_render[self._door_red_pos_trap] = 9

            # Añade el agente
            if 0 <= self._agent_loc[0] < self.height and 0 <= self._agent_loc[1] < self.width:
                grid_para_render[self._agent_loc] = 6

            # Añade la 'T' al char_map
            char_map[9] = "T" # 9 ahora significa Trampa

            # --- FIN DEL HARDCODEO ---

            outfile.write(f"Paso: {self._episode_steps} / {self.max_episode_steps}\n")
            outfile.write(f"Llave: {self._has_key} (0=No, 2=R, 3=B)\n")

            for row in grid_para_render:
                outfile.write("".join([char_map.get(cell, '?') for cell in row]) + "\n")

            return outfile.getvalue()

        elif self._render_mode == "human":
            return None
    
    def _render_frame(self):
        pass
    
    def close(self):
        pass


