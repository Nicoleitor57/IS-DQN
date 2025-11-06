# RL y Estructura de Información en POMDPs

Este repositorio explora los conceptos del paper "On the Role of Information Structure..." (Altabaa & Yang, 2024) aplicando sus ideas teóricas a modelos de caja negra (DRQN/LSTM).

El objetivo es medir cómo diferentes **estructuras de información** (implementadas como entornos POMDP personalizados) desafían la capacidad de un agente DRQN estándar para aprender una política óptima.

---

## 🚀 Configuración y Entorno

Este proyecto utiliza **Python 3.11** y se gestiona con `conda`.

### Instalación

1.  **Clona el repositorio:**
    ```bash
    git clone https://github.com/Nicoleitor57/IS-DQN.git
    cd IS-DQN
    ```

2.  **Crea y activa el entorno de Conda:**
    ```bash
    # Crea el entorno con la versión correcta de Python
    conda create -n rl_info python=3.11
    
    # Activa el entorno
    conda activate rl_info
    ```

3.  **Instala las dependencias:**
    ```bash
    # (Asegúrate de tener un archivo requirements.txt con [gymnasium, ray[rllib], torch])
    pip install -r requirements.txt
    ```

---

## 🔬 Entornos de Prueba

El núcleo de este proyecto son cuatro entornos POMDP diseñados a medida, cada uno para probar una debilidad específica de los modelos de memoria "black-box" (como un LSTM).

### 1. POKeyDoorEnv (El Baseline: POMDP vs. MDP)

* **Concepto Central:** Un agente en un mundo de cuadrícula (grid world) simple con observabilidad parcial.
* **Objetivo:** El agente debe encontrar una llave y luego navegar hasta una puerta.
* **El Desafío (Observabilidad Parcial):** El agente solo ve una cuadrícula de 3x3 a su alrededor.
    * Esto crea **"aliasing de estados"**: múltiples ubicaciones en el mapa (estados latentes $S_t$) se ven idénticas (misma observación $O_t$). Por ejemplo, cualquier pasillo vacío se ve igual.
* **Modelo Puesto a Prueba:**
    * Un **DQN estándar (sin memoria)** fracasará. Es reactivo (solo usa $O_t$) y no puede "recordar" si ya recogió la llave o distinguir pasillos idénticos.
    * Un **DRQN (con LSTM)** debería tener éxito. El LSTM integra el historial de observaciones para construir un "estado de creencia" ($b_t$) que desambigua la situación.

### 2. KeyDoorMazeEnv (El Desafío: Memoria a Largo Plazo)

* **Concepto Central:** Una versión avanzada del `POKeyDoorEnv` diseñada para probar la memoria a largo plazo.
* **Objetivo:** El agente debe recoger una llave específica (ej. **Llave Roja**) y navegar a la puerta del color correspondiente (ej. **Puerta Roja**).
* **El Desafío (Ruido y Distancia):**
    1.  **Memoria de 1-bit:** El agente debe recordar un solo bit crucial (el color de la llave).
    2.  **Distancia:** La llave y las puertas están en extremos opuestos del mapa, forzando un largo viaje (30-50 pasos) a través de pasillos "ruidosos" (observaciones irrelevantes).
* **Modelo Puesto a Prueba:**
    * Este entorno ataca el **desvanecimiento de gradientes (vanishing gradients)**.
    * Un **DRQN (LSTM)** intentará mantener ese "bit de color" en su estado oculto. Sin embargo, es probable que esta información se degrade o se "olvide" después de 50 pasos de procesar observaciones irrelevantes.
    * Se alinea con el paper (Fig. 2d): el estado info-estructural ($\mathcal{I}_h^\dagger$) es mínimo (solo $o_{t-k}$), pero está causalmente distante de la acción $a_t$.

```python
class KeyDoorMazeEnv(gym.Env):
    """
    Entorno de Laberinto con Llave y Puerta Parcialmente Observable (PO-KeyDoor)
    
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
    El estado latente clave es `self._has_key` (0=no, 2=roja, 3=azul).
    Un DRQN debe inferir y *recordar* este estado desde t_llave hasta t_puerta.
    """
