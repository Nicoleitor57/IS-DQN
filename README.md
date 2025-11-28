# IS-DQN: Deep Reinforcement Learning with Information Structure and PSR 🧠

This project investigates the hypothesis that explicitly leveraging **Information Structure (IS)** and **Predictive State Representations (PSR)** as an **inductive bias** can significantly improve the efficiency and stability of agents in **Partially Observable Markov Decision Processes (POMDPs)**.

The design and motivation are grounded in the theoretical framework introduced in the paper *“On the Role of Information Structure in Reinforcement Learning for Partially-Observable Sequential Teams and Games”* (2023), which formalizes how causal information dependencies shape the complexity of sequential decision-making.

---

## 1. Theoretical Background: Information Structure and PSR

### 1.1 Information Structure (IS)

The **Information Structure** of a sequential decision problem describes the causal dependencies among system variables (latent states, observations, and actions).

* **Informationally-Structural State** \( \mathbb{I}_h^\dagger \):  
  Defined as the minimal set of past variables (observable or latent) sufficient to predict future observations.  
  This concept directly quantifies the complexity of the POMDP.

* **PSR and System Dynamics Rank**:  
  Predictive State Representations show that the dynamic rank of the system is upper-bounded by \( |\mathbb{I}_h^\dagger| \).  
  This motivates using PSR-style embeddings as compact, sufficient statistics for RL under partial observability.

---

## 2. Project Scope and Experimental Methodology

The IS-DQN agent is evaluated across three challenging POMDP environments:

| Environment | Description | Partial Observability Challenge |
|------------|-------------|--------------------------------|
| **TwoTigers** | Classic POMDP requiring belief-state tracking. | Bayesian belief updating. |
| **KeyDoorMazeEnv** | Maze navigation with a hidden key and door. | Long-term memory and event dependency. |
| **DelayedStochasticObsEnv** | Observations arrive with random delay. | State inference with stale and noisy information. |

### 2.1 Baseline Agents

IS-DQN is compared against competitive architectures:

* **DQN (Vanilla)** – Standard feedforward Q-network.
* **DRQN** – Recurrent Q-network (LSTM/GRU) for history encoding.
* **PPO-LSTM** – Strong model-based policy gradient method for POMDPs.

---

## 3. Implementation of IS-DQN (Causal-Aware Architecture)

The IS-DQN architecture feeds the Q-network with both the **Bayesian belief state** (the sufficient statistic of the POMDP) and an **explicit uncertainty feature**, enhanced by PER (Prioritized Experience Replay).

### 3.1 IS State Representation

The IS-DQN input state is the concatenated vector:

\[
S^{IS}_t = [b_t, H_t]
\]

Where:

- **\( b_t \)**: Bayesian Belief (probability distribution over latent states).  
  Acts as the PSR-like sufficient statistic of the environment.

- **\( H_t \)**: Normalized Entropy of \( b_t \).  
  Injects explicit uncertainty information, enabling uncertainty-aware decision-making and directed exploration.

---

### 3.2 Custom Q-Network Architecture (QNet)

The QNet is designed with a **dual-branch structure** that maximizes the utility of the Information Structure:

1. **Belief Extractor**  
   Processes the high-dimensional belief vector \( b_t \) into a compact feature representation.

2. **Decision Head**  
   The normalized entropy \( H_t \) is concatenated with the belief features just before the final Q-layer.  
   This ensures that **explicit uncertainty** directly influences action selection.

---

## 4. Ablations and Algorithmic Variants

Ablation studies evaluate the contribution of exploration and sampling mechanisms:

| Ablation | Modification | Objective |
|---------|--------------|-----------|
| **IS-DQN (Full)** | PER + entropy-based exploration | Max performance. |
| **Without PER** | Uniform replay | Value of strategic sampling. |
| **Without Entropy Exploration** | Pure ε-greedy | Value of uncertainty-driven exploration. |
| **Without PER & Entropy** | Base DQN + IS features | Isolate PSR/IS contribution. |

---

## 5. Future Work 🚀

1. Formal estimation of \( |\mathbb{I}_h^\dagger| \) and correlation with optimal PSR dimension.  
2. Framework that allos generalization, bia aproximation of the PSR 
3. Feature-space visualization (UMAP/t-SNE) comparing belief vs PSR embeddings.

# IS-DQN: Deep Reinforcement Learning con Estructura de Información y PSR 🧠

Este proyecto estudia la hipótesis de que incorporar explícitamente la **Estructura de Información (IS)** y las **Representaciones de Estado Predictivas (PSR)** como un **sesgo inductivo** mejora la eficiencia y estabilidad de los agentes en **POMDPs**.

El diseño se fundamenta en el marco teórico introducido en el paper *“On the Role of Information Structure in Reinforcement Learning for Partially-Observable Sequential Teams and Games”* (2023), el cual formaliza cómo la estructura causal de la información determina la complejidad del aprendizaje en entornos parcialmente observables.

---

## 1. Fundamento Teórico: Estructura de Información y PSR

### 1.1 Estructura de Información (IS)

La **Estructura de Información** describe las dependencias causales entre estados latentes, observaciones y acciones.

* **Estado Estructural-Informativo** \( \mathbb{I}_h^\dagger \):  
  Es el conjunto mínimo de variables pasadas suficientes para predecir las observaciones futuras.  
  Este valor determina la complejidad intrínseca del POMDP.

* **PSR y el Rango Dinámico**:  
  Las PSR muestran que el rango dinámico del sistema está acotado por \( |\mathbb{I}_h^\dagger| \), lo que motiva el uso de embeddings compactos de estado predictivo.

---

## 2. Alcance del Proyecto y Metodología Experimental

El IS-DQN se evalúa en tres entornos POMDP exigentes:

| Entorno | Descripción | Desafío |
|---------|-------------|---------|
| **TwoTigers** | POMDP clásico que exige seguimiento de creencias. | Actualización Bayesiana. |
| **KeyDoorMazeEnv** | Laberinto con una llave y una puerta oculta. | Memoria de largo plazo. |
| **DelayedStochasticObsEnv** | Observaciones con retraso estocástico. | Inferencia con información obsoleta. |

### 2.1 Baselines Comparadas

* **DQN (Vanilla)**  
* **DRQN (LSTM/GRU)**  
* **PPO-LSTM** – el baseline más sólido para POMDP.

---

## 3. Implementación del IS-DQN (Arquitectura Causal-Aware)

La implementación del IS-DQN alimenta a la red Q con el **estado de creencia Bayesiano** (el estado suficiente del POMDP) y un **feature explícito de incertidumbre**, junto con optimizaciones de muestreo como PER.

### 3.1 Representación de Estado IS

El vector de entrada para IS-DQN es:

\[
S^{IS}_t = [b_t, H_t]
\]

Donde:

- **\( b_t \)**: Creencia Bayesiana sobre el estado latente.  
  Equivale a la PSR o estado suficiente del sistema.

- **\( H_t \)**: Entropía Normalizada de \( b_t \).  
  Introduce explícitamente la incertidumbre del agente, promoviendo exploración dirigida.

---

### 3.2 Arquitectura de Red Q-Custom (QNet)

La QNet utiliza un diseño de **doble rama**:

1. **Belief Extractor**  
   Procesa el vector de creencia \( b_t \) en un feature compacto.

2. **Decision Head**  
   La entropía normalizada \( H_t \) se concatena con los features del Belief antes de la capa final, garantizando que la **incertidumbre** influya directamente en la acción elegida.

---

## 4. Estudios de Ablación

| Ablación | Cambio | Propósito |
|----------|--------|-----------|
| **IS-DQN (Full)** | PER + exploración por entropía | Máximo rendimiento. |
| **Sin PER** | Replay uniforme | Evaluar muestreo crítico. |
| **Sin Entropía** | Solo ε-greedy | Evaluar exploración dirigida. |
| **Sin PER y sin Entropía** | DQN + estado IS | Aislar el valor de PSR/IS. |

---

## 5. Próximos Pasos 🚀

1. Cuantificar \( |\mathbb{I}_h^\dagger| \) formalmente.  
2. Framework que permita generalizar a otros entornos, mediante aproximacion del psr
3. Visualizar el espacio de estados (UMAP/t-SNE) comparando Belief vs PSR.



