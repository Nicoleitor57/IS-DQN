import numpy as np

# Inicializamos el tensor Z_i (2 estados, 3 acciones, 2 observaciones)
Z_i = np.zeros((2, 3, 2))

# --- 1. Acción = 0 (AE - Escuchar) ---
# Z_i[estado, 0 (AE), observacion]

# s_i = 0 (SL): P(OL|SL,AE)=0.85, P(OR|SL,AE)=0.15
Z_i[0, 0, :] = [0.85, 0.15] 

# s_i = 1 (SR): P(OL|SR,AE)=0.15, P(OR|SR,AE)=0.85
Z_i[1, 0, :] = [0.15, 0.85]

# --- 2. Acción = 1 (AL - Abrir Izquierda) ---
# Z_i[estado, 1 (AL), observacion]
# La observación es aleatoria (50/50), sin importar el estado.

# s_i = 0 (SL): P(OL|SL,AL)=0.5, P(OR|SL,AL)=0.5
Z_i[0, 1, :] = [0.5, 0.5] 

# s_i = 1 (SR): P(OL|SR,AL)=0.5, P(OR|SR,AL)=0.5
Z_i[1, 1, :] = [0.5, 0.5]

# --- 3. Acción = 2 (AR - Abrir Derecha) ---
# Z_i[estado, 2 (AR), observacion]
# La observación también es aleatoria.

# s_i = 0 (SL): P(OL|SL,AR)=0.5, P(OR|SL,AR)=0.5
Z_i[0, 2, :] = [0.5, 0.5] 

# s_i = 1 (SR): P(OL|SR,AR)=0.5, P(OR|SR,AR)=0.5
Z_i[1, 2, :] = [0.5, 0.5]


# Mapeo de índices globales (string) a índices numéricos (int) para Z_i
map_s_i = {'SL': 0, 'SR': 1}
map_a_i = {'AE': 0, 'AL': 1, 'AR': 2}
map_o_i = {'OL': 0, 'OR': 1}

# Mapeo de índices globales (int) a tuplas de strings
# (Tal como los definiste en tu descripción)
map_s = {
    0: ('SL', 'SL'), 1: ('SL', 'SR'), 2: ('SR', 'SL'), 3: ('SR', 'SR')
}
map_a = {
    0: ('AE', 'AE'), 1: ('AE', 'AL'), 2: ('AE', 'AR'),
    3: ('AL', 'AE'), 4: ('AR', 'AE'), 5: ('AL', 'AL'),
    6: ('AL', 'AR'), 7: ('AR', 'AL'), 8: ('AR', 'AR')
}
map_o = {
    0: ('OL', 'OL'), 1: ('OL', 'OR'), 2: ('OR', 'OL'), 3: ('OR', 'OR')
}

# 1. Inicializar el tensor Z final (4 estados, 9 acciones, 4 observaciones)
Z = np.zeros((4, 9, 4))

# 2. Iterar sobre todas las dimensiones del tensor Z
for s_idx in range(4):
    for a_idx in range(9):
        for o_idx in range(4):
            
            # 3. Descomponer los índices globales en sus partes (s1, s2), etc.
            s1_str, s2_str = map_s[s_idx]
            a1_str, a2_str = map_a[a_idx]
            o1_str, o2_str = map_o[o_idx]
            
            # 4. Convertir las partes (strings) a los índices de Z_i (int)
            s1_i = map_s_i[s1_str]
            s2_i = map_s_i[s2_str]
            a1_i = map_a_i[a1_str]
            a2_i = map_a_i[a2_str]
            o1_i = map_o_i[o1_str]
            o2_i = map_o_i[o2_str]
            
            # 5. Aplicar la fórmula de factorización
            # Z(o|s,a) = Z1(o1|s1,a1) * Z2(o2|s2,a2)
            prob_z1 = Z_i[s1_i, a1_i, o1_i]
            prob_z2 = Z_i[s2_i, a2_i, o2_i]
            
            Z[s_idx, a_idx, o_idx] = prob_z1 * prob_z2
            
print("Tensor Z final (4 estados, 9 acciones, 4 observaciones):")
print(Z)

b0 = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)


# Inicializamos el tensor T_i (2 estados, 3 acciones, 2 estados siguientes)
T_i = np.zeros((2, 3, 2))

# --- 1. Acción = 0 (AE - Escuchar) ---
# El estado no cambia.
# s_i = 0 (SL) -> s_i' = 0 (SL)
T_i[0, 0, :] = [1.0, 0.0] 
# s_i = 1 (SR) -> s_i' = 1 (SR)
T_i[1, 0, :] = [0.0, 1.0]

# --- 2. Acción = 1 (AL - Abrir Izquierda) ---
# El entorno se resetea. El nuevo estado es aleatorio (50/50),
# sin importar cuál era el estado anterior.
# s_i = 0 (SL) -> s_i' = (0.5 SL, 0.5 SR)
T_i[0, 1, :] = [0.5, 0.5] 
# s_i = 1 (SR) -> s_i' = (0.5 SL, 0.5 SR)
T_i[1, 1, :] = [0.5, 0.5]

# --- 3. Acción = 2 (AR - Abrir Derecha) ---
# El entorno también se resetea.
# s_i = 0 (SL) -> s_i' = (0.5 SL, 0.5 SR)
T_i[0, 2, :] = [0.5, 0.5] 
# s_i = 1 (SR) -> s_i' = (0.5 SL, 0.5 SR)
T_i[1, 2, :] = [0.5, 0.5]

# Mapeos (los mismos que usamos para Z)
map_s_i = {'SL': 0, 'SR': 1}
map_a_i = {'AE': 0, 'AL': 1, 'AR': 2}
map_s = {0: ('SL', 'SL'), 1: ('SL', 'SR'), 2: ('SR', 'SL'), 3: ('SR', 'SR')}
map_a = {
    0: ('AE', 'AE'), 1: ('AE', 'AL'), 2: ('AE', 'AR'),
    3: ('AL', 'AE'), 4: ('AR', 'AE'), 5: ('AL', 'AL'),
    6: ('AL', 'AR'), 7: ('AR', 'AL'), 8: ('AR', 'AR')
}

# 1. Inicializar el tensor T final (4 estados, 9 acciones, 4 estados siguientes)
# T[s, a, s']
T = np.zeros((4, 9, 4))

# 2. Iterar sobre todas las dimensiones
for s_idx in range(4):
    for a_idx in range(9):
        for s_prime_idx in range(4):
            
            # 3. Descomponer los índices
            s1_str, s2_str = map_s[s_idx]
            a1_str, a2_str = map_a[a_idx]
            s1_p_str, s2_p_str = map_s[s_prime_idx] # s' (estado siguiente)
            
            # 4. Convertir a índices numéricos de T_i
            s1_i = map_s_i[s1_str]
            s2_i = map_s_i[s2_str]
            a1_i = map_a_i[a1_str]
            a2_i = map_a_i[a2_str]
            s1_p_i = map_s_i[s1_p_str]
            s2_p_i = map_s_i[s2_p_str]
            
            # 5. Aplicar la fórmula de factorización
            # T(s'|s,a) = T1(s1'|s1,a1) * T2(s2'|s2,a2)
            prob_t1 = T_i[s1_i, a1_i, s1_p_i]
            prob_t2 = T_i[s2_i, a2_i, s2_p_i]
            
            T[s_idx, a_idx, s_prime_idx] = prob_t1 * prob_t2

def _update_belief(self, b_t, a_idx, o_idx):
    """
    Actualiza el belief state usando el filtro Bayesiano.

    Args:
        b_t (np.array): Belief actual (vector de tamaño 4).
        a_idx (int): Índice de la acción tomada (0-8).
        o_idx (int): Índice de la observación recibida (0-3).
        
    Returns:
        np.array: Nuevo belief b_{t+1} (vector de tamaño 4).
    """

    # --- 1. Predicción (Prediction Step) ---
    # b_hat(s') = Σ_{s} [ T(s' | s, a_t) * b_t(s) ]
    
    # Para la multiplicación matriz-vector, necesitamos T(s' | s, a_t)
    # Nuestro tensor T tiene forma [s, a, s']
    # Así que T_a = self.T[s, a_idx, s']
    # Queremos T_a_transposed = self.T[s, a_idx, :].T, con forma [s', s]
    
    # T_a es la matriz de transición para esta acción: T(s, s' | a_idx)
    T_a = self.T[:, a_idx, :]  # Forma (4, 4) -> (s, s')
    
    # La fórmula es b_hat = T_a.T @ b_t
    # O, b_hat(s') = Σ_{s} T(s, s' | a) * b_t(s)
    # np.einsum('is,s->i', T_a, b_t) # 'is' es T_a, 's' es b_t -> 'i' es b_hat
    # que es lo mismo que:
    b_hat = T_a.T @ b_t  # (4,4).T @ (4,) -> (4,4) @ (4,) -> (4,)
    
    # b_hat es ahora un vector de tamaño 4 (para s')

    # --- 2. Actualización (Update Step) ---
    # b_{t+1}(s') = η * Z(o_{t+1} | s', a_t) * b_hat(s')
    
    # Obtenemos el vector de probabilidades de observación Z(o|s', a)
    # self.Z tiene forma [s', a, o] (asumiendo la construcción anterior)
    Z_o = self.Z[:, a_idx, o_idx]  # Forma (4,) -> Z(o | s', a)
    
    # Multiplicación element-wise (Hadamard)
    b_new_unnormalized = Z_o * b_hat # Forma (4,)
    
    # --- 3. Normalización (Cálculo de η) ---
    # η = 1 / P(o | h_t, a_t)
    # P(o | h_t, a_t) es la suma del vector no normalizado
    prob_obs = np.sum(b_new_unnormalized)
    
    # Añadimos 1e-9 (epsilon) para evitar división por cero
    # si la observación es "imposible" (prob 0).
    b_tplus1 = b_new_unnormalized / (prob_obs + 1e-9)
    
    return b_tplus1







