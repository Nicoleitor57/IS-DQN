# $\text{IS-DQN}$: Reinforcement Learning con Estructura de Información y PSR para POMDPs 🧠

Este proyecto implementa y evalúa varias arquitecturas de **Deep Q-Network (DQN)** que incorporan la **Estructura de Información (IS)** y las **Representaciones de Estado Predictivas (PSR)** como un *sesgo inductivo* explícito. El objetivo es mejorar la eficiencia en la representación de la historia y el rendimiento del agente en entornos de **Proceso de Decisión de Markov Parcialmente Observable (POMDP)**.

***

## 1. Fundamento Teórico: Estructura de Información y PSR

### 1.1 El Marco de la Estructura de Información (IS)

[cite_start]La **Estructura de Información** de un problema de toma de decisiones secuencial describe las dependencias causales entre las variables del sistema[cite: 1]. El *paper* formaliza este concepto a través de:

* [cite_start]**Modelos Generales (POST/POSG):** Propone los modelos **Partially-Observable Sequential Teams (POST)** y **Games (POSG)**, que contienen una representación explícita de la estructura de información[cite: 1].
* [cite_start]**Estado Estructural-Informativo ($\mathbb{I}_{h}^{\dagger}$):** Esta es la cantidad central para el análisis de complejidad[cite: 1]. [cite_start]Se define como el conjunto mínimo de variables pasadas (observables o latentes) que son suficientes para *d-separar* las observaciones pasadas de las observaciones futuras en el grafo causal del sistema[cite: 1].
* [cite_start]**Tractabilidad:** El tamaño de este estado estructural-informativo ($|\mathbb{I}_{h}^{\dagger}|$) caracteriza la complejidad del sistema[cite: 1]. [cite_start]Cuando este tamaño es modesto, el problema se vuelve estadísticamente tratable[cite: 1].

### 1.2 PSR como Representación de Estado Eficaz

[cite_start]Las **Representaciones de Estado Predictivas (PSR)** modelan la dinámica del sistema basándose en la predicción de eventos futuros ("tests") dada la historia pasada, sin modelar explícitamente un estado latente[cite: 1].

* [cite_start]El **rango de la dinámica** de un problema (una medida de su complejidad) está acotado por el tamaño del estado estructural-informativo $|\mathbb{I}_{h}^{\dagger}|$[cite: 1].
* [cite_start]Esto implica que un embedding de PSR (una representación del estado) puede construirse con una dimensión máxima igual a $|\mathbb{I}_{h}^{\dagger}|$, proporcionando una parametrización robusta y eficiente para el aprendizaje[cite: 1].

***

## 2. Implementaciones del Proyecto: Variantes de PSR+DQN

Este repositorio implementa y compara tres arquitecturas DQN diferentes para el entorno **Two-Tigers POMDP**, cada una con un nivel creciente de sesgo inductivo de PSR. [cite_start]Todas las variantes utilizan un cálculo de *oracle Bayesian belief updates* (creencia perfecta) para las posiciones del tigre[cite: 2].

| Variante | Archivo | Características de Entrada (DQN) | Método de Extracción de Features | Énfasis Teórico |
| :--- | :--- | :--- | :--- | :--- |
| **1. Baseline (DQN + Belief)** | `IS_tigers.py` | [cite_start]Vector de creencia de 4-dim (probabilidades por tigre) [cite: 2] | [cite_start]Ninguno [cite: 2] | [cite_start]Comportamiento Base en el espacio de Creencia[cite: 2]. |
| **2. PSR Aprendido (Online)** | `IS_tigers_2.py` | [cite_start]Vector de creencia (4-dim) + 2-dim de predicciones PSR aprendidas [cite: 2] | [cite_start]Red Neuronal (NN) entrenada en línea para predecir $P(O_L \mid listen)$ [cite: 2] | [cite_start]Aproximación simple de PSR basada en tests[cite: 2]. |
| **3. PSR Espectral (Offline)** | `IS_tigers_3.py` | [cite_start]Vector de creencia (4-dim) + $r$-dim de embedding PSR (default $r=2$) [cite: 2] | [cite_start]Descomposición SVD de la matriz Historia $\times$ Test ($P_{HT}$) recolectada en una fase de *warmup* [cite: 2] | [cite_start]Aplicación directa de la teoría PSR espectral[cite: 2]. |

### Componentes Clave del Código
* [cite_start]**`IS/DQN.py`**: Contiene las clases base del agente DQN, incluyendo `QNet` y `ReplayBuffer`[cite: 3].
* **Entornos**: Los entornos POMDP utilizados (como el `TwoTigersEnv.py`) son esenciales para probar la robustez de las representaciones de estado en condiciones de observación parcial.

***

## 3. Próximos Pasos / Extensiones Sugeridas 🚀

1.  **Evaluación Estadística con Múltiples Semillas:**
    * Ejecutar cada variante con **5–10 semillas** y reportar la media ± desviación estándar o intervalo de confianza para validar la significancia estadística.
2.  **Integración de Arquitecturas Recurrentes (DRQN):**
    * [cite_start]Reemplazar la Q-Network *feedforward* con una red recurrente (GRU/LSTM) e incorporar los features PSR[cite: 2].
3.  **Análisis de Sensibilidad a la $\alpha$-Robustez:**
    * Evaluar cómo la elección del rango $r$ en la PSR espectral afecta la estabilidad y el rendimiento del DQN.
4.  **Extensión a Estructuras de Información Alternativas:**
    * [cite_start]Aplicar los métodos PSR-DQN a otros entornos con estructuras de información conocidas (ej. memoria limitada)[cite: 2].

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
