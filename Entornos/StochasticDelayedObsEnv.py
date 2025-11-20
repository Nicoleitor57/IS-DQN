import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque 

from Entornos.StochasticGridEnv import StochasticGridEnv

# NOTA: La clase StochasticGridEnv debe estar definida en tu script o ser importada.

class DelayedStochasticObsEnv(gym.Env):
    """
    Entorno con retardo de observación.
    
    Envuelve StochasticGridEnv (el que tiene el slip_prob) y devuelve el 
    estado completo (x, y, has_key) de hace 'delay_steps' pasos.
    
    La observación es el vector [x, y, k], compatible con el filtro de Propagación.
    """
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, size=10, max_episode_steps=100, render_mode="ansi", 
                 slip_prob=0.1, delay_steps=3):
        
        self.delay_steps = delay_steps 
        
        # 1. Inicializar el entorno estocástico base
        self.base_env = StochasticGridEnv( 
            size=size, 
            max_episode_steps=max_episode_steps, 
            render_mode=render_mode, 
            slip_prob=slip_prob
        )
        
        # 2. Definir el espacio de observación para el filtro (x, y, k)
        # El estado k (has_key) es 0 o 1.
        self.observation_space = spaces.Box(
            low=0, 
            high=size, 
            shape=(3,), # [x_pasado, y_pasado, k_pasado]
            dtype=np.int8
        )
        self.action_space = self.base_env.action_space
        
        # 3. Historial de estados (NO observaciones 3x3)
        self.state_history = deque(maxlen=delay_steps + 1)
        self.action_history = deque(maxlen=delay_steps)
        self._current_steps = 0

    def _get_current_state_vector(self):
        """Convierte el estado interno del base_env a un vector [x, y, k]."""
        x, y = self.base_env._agent_location
        # Usamos 1 para True, 0 para False para el filtro
        k_val = 1 if self.base_env._has_key else 0 
        return np.array([x, y, k_val], dtype=np.int8)


    def reset(self, seed=None, options=None):
        # 1. Resetear el entorno base (obtiene s_0)
        _, info_base = self.base_env.reset(seed=seed, options=options)
        
        # 2. Obtener el estado inicial (s_0)
        initial_state = self._get_current_state_vector()
        
        # 3. Inicializar el historial
        self.state_history.clear()
        self.action_history.clear()
        
        # Llenar el historial con s_0 (k+1 veces)
        for _ in range(self.delay_steps + 1):
            self.state_history.append(initial_state)
        
        self._current_steps = 0
        
        # 4. La primera observación es s_0 (compatible con s_{t-k})
        observation = self.state_history[0] 
        
        return observation, {"delay_info": "Observation is s_0", **info_base}

    def step(self, action):
        
        # 1. Ejecutar la acción en el entorno base
        _, reward, terminated, truncated, info_base = self.base_env.step(action)
        
        # 2. Registrar el estado actual (s_t)
        current_state = self._get_current_state_vector()
        
        # 3. Obtener la observación (s_{t-k}) del historial
        past_state = self.state_history.popleft() # s_{t-k}
        
        # 4. Actualizar historiales
        self.state_history.append(current_state) # s_t se añade
        self.action_history.append(action)       # a_t se añade
        self._current_steps += 1

        # 5. La observación DEVUELTA es el estado completo pasado
        observation = past_state
        
        delay_info = {
            "delay_info": f"Observation is s_{self._current_steps - self.delay_steps}",
            "past_state_obs": past_state
        }
        
        return observation, reward, terminated, truncated, {**info_base, **delay_info}

    def render(self):
        return self.base_env.render()

    def close(self):
        return self.base_env.close()