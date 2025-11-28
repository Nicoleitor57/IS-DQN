# $\text{IS-DQN}$: Reinforcement Learning with Information Structure and PSR for POMDPs 🧠

This project implements and evaluates several **Deep Q-Network (DQN)** architectures that incorporate **Information Structure (IS)** and **Predictive State Representations (PSR)** as an explicit inductive bias. The theoretical motivation comes from the paper **“On the Role of Information Structure in Reinforcement Learning for Partially-Observable Sequential Teams and Games”**, which formalizes how information flow and causal structure determine the statistical complexity of learning in POMDPs.

***

## 1. Theoretical Foundation: Information Structure and PSR

### 1.1 Information Structure (IS)

The above paper introduces a unified view of information flow in sequential decision-making problems. In particular, it defines:

* **General Models (POST/POSG):**  
  The paper introduces **Partially-Observable Sequential Teams (POST)** and **Partially-Observable Stochastic Games (POSG)** as general frameworks that explicitly encode the information structure available to decision-makers.

* **Information-Structural State ($\mathbb{I}_{h}^{\dagger}$):**  
  A key contribution of the paper is the definition of the *information-structural state*, the minimal subset of past variables needed to d-separate past and future observations in the causal graph.  
  This object characterizes the “informational complexity” of a POMDP-like system.

* **Tractability:**  
  The paper proves that when the size of this state ($|\mathbb{I}_{h}^{\dagger}|$) is small, reinforcement learning becomes statistically tractable despite partial observability.

### 1.2 Predictive State Representations (PSR)

The project also draws on the paper’s observation that the **dynamical rank** of a system is bounded by the size of the information-structural state. This directly implies that:

* A PSR embedding only needs dimension up to $|\mathbb{I}_{h}^{\dagger}|$.  
* PSRs provide an efficient and theoretically grounded state representation for POMDP RL.

This motivates the use of PSR-enhanced DQN architectures in this repository.

***

## 2. Implemented Architectures: PSR+DQN Variants

We evaluate three DQN variants on the classic **Two-Tigers POMDP**, each incorporating different degrees of structure inspired by the paper.

All models use **oracle Bayesian belief updates** for the tiger positions.

| Variant | File | DQN Input | Feature Method | Motivation |
| :--- | :--- | :--- | :--- | :--- |
| **1. Baseline (DQN + Belief)** | `IS_tigers.py` | 4-dim belief vector | None | Standard belief-state approach. |
| **2. Learned PSR (Online)** | `IS_tigers_2.py` | Belief + 2-dim learned PSR tests | Online NN predicting future observations | Inspired by low-rank predictive modeling. |
| **3. Spectral PSR (Offline)** | `IS_tigers_3.py` | Belief + $r$-dim spectral PSR | SVD on History × Test matrix | Motivated by PSR theory and the paper’s rank bounds. |

### Relevant Code Files

* **`IS/DQN.py`**: Core DQN implementation (`QNet`, `ReplayBuffer`, etc.).
* **`TwoTigersEnv.py`**: POMDP environment used for evaluation under partial observability.

***

## 3. Suggested Extensions 🚀

1. **Multiple Seeds & Confidence Intervals**  
   Evaluate each architecture using several seeds to ensure statistical significance.

2. **Recurrent Architectures (DRQN)**  
   Replace the feedforward Q-network with an LSTM/GRU and complement it with PSR features.

3. **Varying PSR Rank ($r$)**  
   Study how PSR dimensionality influences stability and generalization.

4. **Other Information-Structure Settings**  
   Following the paper, test environments where the information structure is known or can be restricted.

***

## 4. Running the Code

Dependencies: `torch`, `numpy`, `gymnasium`, `tensorboard`.

### Execute the Three Variants

```bash
# 1. Baseline: DQN + Oracle Belief
python -m IS.IS_tigers

# 2. Online Learned PSR + DQN
python -m IS.IS_tigers_2

# 3. Spectral PSR (Offline SVD) + DQN
python -m IS.IS_tigers_3
```



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
