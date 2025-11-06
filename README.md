# RL y Estructura de Información en POMDPs

Este repositorio investiga cómo la **estructura de información**, un concepto clave del paper "[On the Role of Information Structure...](https://arxiv.org/abs/2306.02243)" (Altabaa & Yang, NeurIPS 2024), puede ser utilizada para **mejorar positivamente el rendimiento** de agentes de Deep Q-Networks (DQN) en entornos parcialmente observables (POMDPs).

El objetivo es demostrar que un agente estándar de "caja negra" como un DRQN (DQN + LSTM) es estadísticamente ineficiente. Al inyectar conocimiento sobre la estructura causal del problema (como un "sesgo inductivo"), podemos lograr un aprendizaje drásticamente más rápido y robusto.

Para probar esta tesis, se han diseñado cuatro entornos POMDP personalizados.

---

## 🚀 Configuración y Entorno

Este proyecto utiliza **Python 3.11** y se gestiona con `conda`.

### Instalación

1.  **Clona el repositorio:**
    ```bash
    git clone [https://github.com/Nicoleitor57/IS-DQN.git](https://github.com/Nicoleitor57/IS-DQN.git)
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
    # Instala las bibliotecas requeridas
    pip install -r requirements.txt
    ```

---

## 🔬 Entornos de Prueba

El núcleo de este proyecto son cuatro entornos POMDP diseñados a medida, cada uno para probar una debilidad específica de los modelos de memoria "black-box".

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
```
### 3. TwoTigersEnv (El Desafío: Estados Factoreados)

* **Concepto Central:** El agente debe resolver dos problemas de POMDP (el clásico "Juego del Tigre") de forma independiente y simultánea.
* **Objetivo del Agente:** En cada paso, el agente recibe dos observaciones de sonido (una para el Tigre 1, otra para el Tigre 2) y debe tomar dos acciones (una para cada juego: "escuchar", "abrir izquierda" o "abrir derecha"). La recompensa es la suma de los resultados. Los dos juegos son estadísticamente independientes.
* **El Desafío (Factorización):** El desafío es la **factorización**. El estado latente $S$ está perfectamente dividido en dos componentes independientes: $S = (S_1, S_2)$. Un agente óptimo debe mantener dos "estados de creencia" separados ($b_1$ y $b_2$).
* **Modelo Puesto a Prueba:** Este entorno prueba si un modelo *black-box* aprenderá **correlaciones espurias**.
    * Un **DRQN (LSTM) monolítico** recibirá el vector de observación `[obs_1, obs_2]` y lo procesará con un solo estado oculto. Inevitablemente, aprenderá falsas correlaciones (ej. "si oigo al Tigre 1 a la izquierda, es más probable que el Tigre 2 esté a la derecha", aunque sea falso).
    * Se alinea con el paper (Fig. 2a/b): un modelo que conoce el DAG sabría que el problema está factoreado. Un modelo con **dos LSTMs separados** (uno para cada tigre) debería aprender mucho más rápido.

```python
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
```

---

### 4. DelayedObsEnv (El Desafío: Estructura Causal Compleja)

* **Concepto Central:** Un grid world estándar, pero con una estructura de información no-Markoviana: la observación del agente está retrasada.
* **Objetivo del Agente:** Similar al `POKeyDoorEnv`, navegar por el laberinto para alcanzar una meta.
* **El Desafío (No-Markoviano):** Este **no es un POMDP estándar**. En un POMDP, la observación actual depende del estado actual ($O_t \sim P(S_t)$). Aquí, la observación que el agente recibe en el tiempo $t$ es una observación de su estado de hace $m$ pasos ($O_t \sim P(S_{t-m})$).
* **Modelo Puesto a Prueba:** Este es el desafío más complejo. El agente no puede confiar solo en su memoria de observaciones; **debe usar su memoria de acciones**.
    * Para inferir su estado actual $S_t$, el agente debe tomar su observación retrasada $O_t$ (que le dice dónde estaba en $S_{t-m}$) y simular mentalmente las $m$ acciones que ha tomado desde entonces ($A_{t-m}, \dots, A_{t-1}$) para proyectar su posición actual.
    * Un **DRQN estándar** fracasará casi con seguridad, ya que el LSTM solo recibe observaciones y no está explícitamente diseñado para integrar su propio historial de acciones de esta manera.
 

---

## 🎯 Hoja de Ruta (To-Do)

* [x] 1. Implementar Baselines (DRQN estándar en todos los entornos).
* [ ] 2. **(KeyDoorMazeEnv)** Implementar "Sesgo Inductivo":
    * Crear un entorno "privilegiado" donde el vector de observación `obs` incluya `has_key` (ej. `[obs_3x3, has_key]`).
    * Comparar la eficiencia de muestreo contra el baseline de caja negra.
* [ ] 3. **(TwoTigersEnv)** Implementar Modelo Factoreado:
    * Crear un modelo personalizado de RLlib con *dos LSTMs separados*.
    * El input `[obs_1, obs_2]` se debe dividir; `obs_1` va al `LSTM_1` y `obs_2` al `LSTM_2`.
    * Comparar contra el baseline de LSTM monolítico para medir las correlaciones espurias.
* [ ] 4. **(DelayedObsEnv)** Implementar Modelo Causal:
    * Crear un modelo personalizado que reciba el historial de acciones.
    * El input al LSTM debe ser `[O_t, A_{t-1}, A_{t-m}]`.
    * Probar si el agente puede aprender a "simular" su estado futuro.
* [ ] 5. **Análisis Final:**
    * Redactar el análisis final comparando el rendimiento del DRQN *black-box* contra los modelos con "estructura de información" explícita.
