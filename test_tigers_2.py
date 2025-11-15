import gymnasium as gym
import numpy as np
# Asumimos que tu entorno se llama TwoTigersEnv y está en TwoTigersEnv.py
from Entornos.TwoTigersEnv import TwoTigersEnv 

class BeliefStateWrapper(gym.Wrapper):
    """
    Este wrapper transforma un entorno POMDP (como TwoTigersEnv) en un MDP 
    totalmente observable, donde la "observación" es el belief state (PSR).
    
    Asume que el entorno base (env):
    1. Tiene un espacio de acciones: gym.spaces.Discrete(9)
    2. Devuelve observaciones: gym.spaces.Discrete(4)
    """

    def __init__(self, env):
        super().__init__(env)
        
        # --- 1. Definir constantes del POMDP ---
        self.num_states = 4  # |S| = 4  [(SL,SL), (SL,SR), (SR,SL), (SR,SR)]
        self.num_actions = 9 # |A| = 9  [(AE,AE), (AE,AL), ..., (AR,AR)]
        self.num_obs = 4     # |Ω| = 4  [(OL,OL), (OL,OR), (OR,OL), (OR,OR)]

        # --- 2. Construir Modelos (T y Z) ---
        # (Estos se basan en todo lo que discutimos anteriormente)
        self.T = self._build_transition_model() # Matriz [s, a, s']
        self.Z = self._build_observation_model()  # Matriz [s', a, o]
        
        # --- 3. Definir Belief Inicial (b0) ---
        self.b0 = np.full(self.num_states, 1.0 / self.num_states, dtype=np.float32)
        
        # Guardar el belief actual
        self.current_belief = self.b0.copy()

        # --- 4. Sobrescribir el Espacio de Observación ---
        # El agente DQN ya no verá el gym.spaces.Discrete(4).
        # Verá un vector de flotantes (el belief) de tamaño |S|.
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.num_states,),
            dtype=np.float32
        )

    def reset(self, **kwargs):
        """
        Resetea el entorno real y también resetea el belief state 
        a la distribución inicial b0.
        """
        # Resetea el entorno subyacente. 
        # Ignoramos la 'obs' inicial, ya que b0 es nuestra "obs" inicial.
        obs, info = self.env.reset(**kwargs)
        
        # Resetea el belief a b0
        self.current_belief = self.b0.copy()
        
        # Devuelve el belief state como la primera observación
        return self.current_belief, info

    def step(self, action):
        """
        Toma una acción, obtiene la (obs, recompensa) real, actualiza
        el belief, y devuelve el *nuevo* belief como el estado.
        """
        # 1. 'action' es el a_idx (0-8) elegido por el agente DQN
        
        # 2. Ejecuta el paso en el entorno real
        # 'obs' será la observación real (p.ej., 0, 1, 2, o 3)
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # 3. Mapea la observación real a su índice (si es necesario)
        #    Asumimos que 'obs' ya es el o_idx (0-3)
        o_idx = self._map_obs_to_index(obs)
        
        # 4. Actualiza el belief state usando el filtro Bayesiano
        b_t = self.current_belief
        a_idx = action
        
        b_tplus1 = self._update_belief(b_t, a_idx, o_idx)
        
        # 5. Guarda el nuevo belief
        self.current_belief = b_tplus1
        
        # 6. Devuelve el *nuevo belief* como el "estado" al agente
        return self.current_belief, reward, terminated, truncated, info

    # --- Motor del Filtro Bayesiano ---

    def _update_belief(self, b_t, a_idx, o_idx):
        """
        Actualiza el belief state usando el filtro Bayesiano.
        b_{t+1}(s') = η * Z(o | s', a) * Σ_{s} [ T(s' | s, a) * b_t(s) ]
        """
        
        # --- 1. Predicción (Prediction Step) ---
        # b_hat(s') = Σ_{s} [ T(s' | s, a_t) * b_t(s) ]
        #
        # T_a tiene forma [s, s'] para el a_idx dado
        T_a = self.T[:, a_idx, :]  # Matriz T(s, s' | a) de forma (4, 4)
        
        # Multiplicación Matriz-Vector: b_hat = T_a.T @ b_t
        # T_a.T tiene forma (s', s). b_t tiene forma (s,).
        # El resultado b_hat tiene forma (s',).
        b_hat = T_a.T @ b_t
        
        # --- 2. Actualización (Update Step) ---
        # b_{t+1}(s') = η * Z(o_{t+1} | s', a_t) * b_hat(s')
        
        # Z_o es el vector de probabilidades Z(o | s', a)
        # self.Z tiene forma [s', a, o]
        Z_o = self.Z[:, a_idx, o_idx] # Vector de forma (4,)
        
        # Multiplicación element-wise (Hadamard)
        b_new_unnormalized = Z_o * b_hat
        
        # --- 3. Normalización (Cálculo de η) ---
        prob_obs = np.sum(b_new_unnormalized)
        
        if prob_obs < 1e-9:
            # Ocurrió una observación "imposible" (prob 0).
            # Esto no debería pasar si el modelo es correcto.
            # Como fallback, reseteamos a una creencia uniforme.
            return self.b0.copy()
        else:
            b_tplus1 = b_new_unnormalized / prob_obs
            return b_tplus1

    def _map_obs_to_index(self, obs):
        """
        Función de ayuda para convertir la observación del entorno
        a un índice (0-3).
        
        Esta función ahora maneja dos casos:
        1. El env devuelve un int (0-3) -> lo retorna.
        2. El env devuelve un array/lista (p.ej., [0, 0]) -> lo convierte.
        """
        
        

        # Caso 1: La observación es un array/lista, p.ej., [0, 0] o np.array([0, 0])
        # Asumimos que [o1, o2] donde o1, o2 son 0 (OL) o 1 (OR)
        if isinstance(obs, (list, np.ndarray)) and len(obs) == 2:
            o1, o2 = obs
            
            # Convertir [o1, o2] a un índice único usando base-2
            # [0, 0] -> 0*2 + 0 = 0  (Corresponde a 'OL', 'OL')
            # [0, 1] -> 0*2 + 1 = 1  (Corresponde a 'OL', 'OR')
            # [1, 0] -> 1*2 + 0 = 2  (Corresponde a 'OR', 'OL')
            # [1, 1] -> 1*2 + 1 = 3  (Corresponde a 'OR', 'OR')
            o_idx = int(o1) * 2 + int(o2)
            
            if 0 <= o_idx <= 3:
                return o_idx
            else:
                # Esto pasaría si obs fuera algo como [2, 0], lo cual es inválido
                raise ValueError(f"Observación (array) con valores inválidos: {obs}.")

        # Caso 2: La observación ya es el índice (0-3)
        if isinstance(obs, (int, np.integer)): # Ser más robusto con tipos de numpy
            if 0 <= obs <= 3:
                return obs
            else:
                raise ValueError(f"Observación (int) fuera de rango: {obs}. Se esperaba 0-3.")

        
            
        # Si no es ni un int 0-3 ni un array de 2 elementos
        raise ValueError(f"Formato de observación no reconocida: {obs}. Se esperaba un int 0-3 o un array/lista de 2 elementos.")

    # --- Funciones de Construcción de Modelos (Helpers) ---

    def _build_observation_model(self):
        """
        Construye el tensor Z(s', a, o) = P(o | s', a) de 4x9x4.
        """
        # Mapeos (solo para construcción)
        map_s_i = {'SL': 0, 'SR': 1}
        map_a_i = {'AE': 0, 'AL': 1, 'AR': 2}
        map_o_i = {'OL': 0, 'OR': 1}
        map_s = {0: ('SL', 'SL'), 1: ('SL', 'SR'), 2: ('SR', 'SL'), 3: ('SR', 'SR')}
        map_a = {
            0: ('AE', 'AE'), 1: ('AE', 'AL'), 2: ('AE', 'AR'),
            3: ('AL', 'AE'), 4: ('AR', 'AE'), 5: ('AL', 'AL'),
            6: ('AL', 'AR'), 7: ('AR', 'AL'), 8: ('AR', 'AR')
        }
        map_o = {0: ('OL', 'OL'), 1: ('OL', 'OR'), 2: ('OR', 'OL'), 3: ('OR', 'OR')}

        # 1. Tensor Z_i base (2x3x2)
        Z_i = np.zeros((2, 3, 2))
        Z_i[0, 0, :] = [0.85, 0.15] # P(o|s=SL, a=AE)
        Z_i[1, 0, :] = [0.15, 0.85] # P(o|s=SR, a=AE)
        Z_i[:, 1, :] = 0.5         # P(o|s=*, a=AL) = 0.5
        Z_i[:, 2, :] = 0.5         # P(o|s=*, a=AR) = 0.5

        # 2. Tensor Z final (4x9x4) -> Z[s', a, o]
        Z = np.zeros((self.num_states, self.num_actions, self.num_obs))
        
        for s_prime_idx in range(self.num_states):
            for a_idx in range(self.num_actions):
                for o_idx in range(self.num_obs):
                    
                    s1_str, s2_str = map_s[s_prime_idx]
                    a1_str, a2_str = map_a[a_idx]
                    o1_str, o2_str = map_o[o_idx]
                    
                    s1_i = map_s_i[s1_str]
                    s2_i = map_s_i[s2_str]
                    a1_i = map_a_i[a1_str]
                    a2_i = map_a_i[a2_str]
                    o1_i = map_o_i[o1_str]
                    o2_i = map_o_i[o2_str]
                    
                    prob_z1 = Z_i[s1_i, a1_i, o1_i]
                    prob_z2 = Z_i[s2_i, a2_i, o2_i]
                    
                    Z[s_prime_idx, a_idx, o_idx] = prob_z1 * prob_z2
        
        return Z

    def _build_transition_model(self):
        """
        Construye el tensor T(s, a, s') = P(s' | s, a) de 4x9x4.
        """
        # Mapeos
        map_s_i = {'SL': 0, 'SR': 1}
        map_a_i = {'AE': 0, 'AL': 1, 'AR': 2}
        map_s = {0: ('SL', 'SL'), 1: ('SL', 'SR'), 2: ('SR', 'SL'), 3: ('SR', 'SR')}
        map_a = {
            0: ('AE', 'AE'), 1: ('AE', 'AL'), 2: ('AE', 'AR'),
            3: ('AL', 'AE'), 4: ('AR', 'AE'), 5: ('AL', 'AL'),
            6: ('AL', 'AR'), 7: ('AR', 'AL'), 8: ('AR', 'AR')
        }

        # 1. Tensor T_i base (2x3x2) -> T_i[s, a, s']
        T_i = np.zeros((2, 3, 2))
        T_i[0, 0, :] = [1.0, 0.0] # (s=SL, a=AE) -> s'=SL
        T_i[1, 0, :] = [0.0, 1.0] # (s=SR, a=AE) -> s'=SR
        T_i[:, 1, :] = 0.5         # (s=*, a=AL) -> s' ~ Unif(0.5, 0.5)
        T_i[:, 2, :] = 0.5         # (s=*, a=AR) -> s' ~ Unif(0.5, 0.5)

        # 2. Tensor T final (4x9x4) -> T[s, a, s']
        T = np.zeros((self.num_states, self.num_actions, self.num_states))
        
        for s_idx in range(self.num_states):
            for a_idx in range(self.num_actions):
                for s_prime_idx in range(self.num_states):
                    
                    s1_str, s2_str = map_s[s_idx]
                    a1_str, a2_str = map_a[a_idx]
                    s1_p_str, s2_p_str = map_s[s_prime_idx]
                    
                    s1_i = map_s_i[s1_str]
                    s2_i = map_s_i[s2_str]
                    a1_i = map_a_i[a1_str]
                    a2_i = map_a_i[a2_str]
                    s1_p_i = map_s_i[s1_p_str]
                    s2_p_i = map_s_i[s2_p_str]
                    
                    prob_t1 = T_i[s1_i, a1_i, s1_p_i]
                    prob_t2 = T_i[s2_i, a2_i, s2_p_i]
                    
                    T[s_idx, a_idx, s_prime_idx] = prob_t1 * prob_t2
        
        return T
    
    
    
#test clase BeliefStateWrapper

print("Probando BeliefStateWrapper...")
env = TwoTigersEnv(max_episode_steps=1000)
wrapped_env = BeliefStateWrapper(env)
print(wrapped_env._build_observation_model())
print(wrapped_env._build_transition_model())
# print("\n--- Probando el 'reset' y 'step' del wrapper ---")
# try:
#     print("Probando wrapped_env.reset()...")
#     obs, info = wrapped_env.reset()
#     print(f"Observación inicial (belief): {obs.shape}")

#     print("\nProbando wrapped_env.step(0)...")
#     # Tomamos una acción de ejemplo (índice 0 = 'AE', 'AE')
#     action = 0 
#     new_obs, reward, terminated, truncated, info = wrapped_env.step(action)
#     print("¡El step funcionó!")

# except ValueError as e:
#     print("\n¡FALLÓ! Se replicó el error:")
#     print(e)


from stable_baselines3 import DQN
model = DQN("MlpPolicy", wrapped_env, verbose=1)
model.learn(total_timesteps=300000)