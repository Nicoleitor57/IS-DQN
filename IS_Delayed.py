
import gymnasium as gym
import numpy as np
import pandas as pd
import time
import os
import glob
import matplotlib.pyplot as plt
import random
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


try:
    from Entornos.StocasticDelayedObsEnv import StochasticGridEnv
except ImportError:
    print("ERROR CRÍTICO: No se encuentra 'Entornos/KeyDoorMazeEnv.py'.")
    exit()



# ==============================================================================
# 1. WRAPPER: BELIEF STATE (PSR) + ENTROPY
# ==============================================================================
class KeyDoorBeliefWrapper(gym.Wrapper):
    """
    Wrapper que convierte las observaciones parciales en un estado de creencia
    basado en un histograma de observaciones recientes.
    Además, añade una componente de entropía al estado.
    """
   
   def __init__(self, env, hist_size=3):
       super ().__init__(env)
       self.base_env = env.unwrapped
       self.H = self.base_env.height
       self.W = self.base_env.width
       
       