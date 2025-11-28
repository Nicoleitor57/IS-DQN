# $\text{IS-DQN}$: Reinforcement Learning con Estructura de Información y PSR para POMDPs 🧠

Este proyecto implementa y evalúa varias arquitecturas de **Deep Q-Network (DQN)** que incorporan la **Estructura de Información (IS)** y las **Representaciones de Estado Predictivas (PSR)** como un *sesgo inductivo* explícito. El objetivo es mejorar la eficiencia en la representación de la historia y el rendimiento del agente en entornos de **Proceso de Decisión de Markov Parcialmente Observable (POMDP)**.

***

## 1. Fundamento Teórico: Estructura de Información y PSR

### 1.1 El Marco de la Estructura de Información (IS)

La **Estructura de Información** de un problema de toma de decisiones secuencial describe las dependencias causales entre las variables del sistema. El *paper* formaliza este concepto a través de:

* **Modelos Generales (POST/POSG):** Propone los modelos **Partially-Observable Sequential Teams (POST)** y **Games (POSG)**, que contienen una representación explícita de la estructura de información.
* **Estado Estructural-Informativo ($\mathbb{I}_{h}^{\dagger}$):** Esta es la cantidad central para el análisis de complejidad. Se define como el conjunto mínimo de variables pasadas (observables o latentes) que son suficientes para *d-separar* las observaciones pasadas de las observaciones futuras en el grafo causal del sistema.
* **Tractabilidad:** El tamaño de este estado estructural-informativo ($|\mathbb{I}_{h}^{\dagger}|$) caracteriza la complejidad del sistema. Cuando este tamaño es modesto, el problema se vuelve estadísticamente tratable.

### 1.2 PSR como Representación de Estado Eficaz

Las **Representaciones de Estado Predictivas (PSR)** modelan la dinámica del sistema basándose en la predicción de eventos futuros ("tests") dada la historia pasada, sin modelar explícitamente un estado latente.

* El **rango de la dinámica** de un problema (una medida de su complejidad) está acotado por el tamaño del estado estructural-informativo $|\mathbb{I}_{h}^{\dagger}|$.
* Esto implica que un embedding de PSR (una representación del estado) puede construirse con una dimensión máxima igual a $|\mathbb{I}_{h}^{\dagger}|$, proporcionando una parametrización robusta y eficiente para el aprendizaje.

***

## 2. Implementaciones del Proyecto: Variantes de PSR+DQN

Este repositorio implementa y compara tres arquitecturas DQN diferentes para el entorno **Two-Tigers POMDP**, cada una con un nivel creciente de sesgo inductivo de PSR. Todas las variantes utilizan un cálculo de *oracle Bayesian belief updates* (creencia perfecta) para las posiciones del tigre.

| Variante | Archivo | Características de Entrada (DQN) | Método de Extracción de Features | Énfasis Teórico |
| :--- | :--- | :--- | :--- | :--- |
| **1. Baseline (DQN + Belief)** | `IS_tigers.py` | Vector de creencia de 4-dim (probabilidades por tigre) | Ninguno | Comportamiento Base en el espacio de Creencia. |
| **2. PSR Aprendido (Online)** | `IS_tigers_2.py` | Vector de creencia (4-dim) + 2-dim de predicciones PSR aprendidas | Red Neuronal (NN) entrenada en línea para predecir $P(O_L \mid listen)$ | Aproximación simple de PSR basada en tests. |
| **3. PSR Espectral (Offline)** | `IS_tigers_3.py` | Vector de creencia (4-dim) + $r$-dim de embedding PSR (default $r=2$) | Descomposición SVD de la matriz Historia × Test ($P_{HT}$) recolectada en una fase de *warmup* | Aplicación directa de la teoría PSR espectral. |

### Componentes Clave del Código
* **`IS/DQN.py`**: Contiene las clases base del agente DQN, incluyendo `QNet` y `ReplayBuffer`.
* **Entornos**: Los entornos POMDP utilizados (como el `TwoTigersEnv.py`) son esenciales para probar la robustez de las representaciones de estado en condiciones de observación parcial.

***

## 3. Próximos Pasos / Extensiones Sugeridas 🚀

1. **Evaluación Estadística con Múltiples Semillas:**
   * Ejecutar cada variante con **5–10 semillas** y reportar la media ± desviación estándar o intervalo de confianza para validar la significancia estadística.

2. **Integración de Arquitecturas Recurrentes (DRQN):**
   * Reemplazar la Q-Network *feedforward* con una red recurrente (GRU/LSTM) e incorporar los features PSR.

3. **Análisis de Sensibilidad a la $\alpha$-Robustez:**
   * Evaluar cómo la elección del rango $r$ en la PSR espectral afecta la estabilidad y el rendimiento del DQN.

4. **Extensión a Estructuras de Información Alternativas:**
   * Aplicar los métodos PSR-DQN a otros entornos con estructuras de información conocidas (ej. memoria limitada).

***

## 4. Uso y Ejecución

Asegúrate de tener instaladas las dependencias (`torch`, `numpy`, `gymnasium`, `tensorboard`).

### Ejecución de las Variantes Two-Tigers

```bash
# 1. Baseline: DQN + Oracle Belief
python -m IS.IS_tigers

# 2. PSR Aprendido (Online) + DQN
python -m IS.IS_tigers_2

# 3. PSR Espectral (Offline SVD) + DQN
python -m IS.IS_tigers_3
