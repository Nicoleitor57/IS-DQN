from Entornos.PODoorEnv import POKeyDoorEnv
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class DelayedObsEnv(POKeyDoorEnv): # Hereda del anterior
    
    # 1. Aceptar max_episode_steps en __init__
    def __init__(self, size=10, delay_steps=3, max_episode_steps=100, render_mode="ansi"):
        # 2. Pasarlo al __init__ de la clase padre (POKeyDoorEnv)
        super().__init__(
            size=size, 
            max_episode_steps=max_episode_steps, 
            render_mode=render_mode
        )
        
        self.delay_steps = delay_steps
        self._obs_buffer = []

    def reset(self, seed=None, options=None):
        # Esta función ya es correcta.
        # super().reset() reinicia el self._current_steps del padre.
        base_obs, info = super().reset(seed=seed)
        
        empty_obs = np.ones((3, 3), dtype=np.int8) * 1 
        self._obs_buffer = [empty_obs] * self.delay_steps
        
        obs_to_return = self._obs_buffer.pop(0)
        self._obs_buffer.append(base_obs)

        return obs_to_return, info

    def step(self, action):
        # Esta función ya es correcta.
        # super().step() incrementa self._current_steps y calcula
        # 'truncated' por nosotros.
        _, reward, terminated, truncated, info = super().step(action)
        
        obs_to_return = self._obs_buffer.pop(0)
        true_obs_now = super()._get_obs() 
        self._obs_buffer.append(true_obs_now)

        # Devolvemos el 'truncated' que recibimos del padre
        return obs_to_return, reward, terminated, truncated, info