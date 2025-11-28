# $\text{IS-DQN}$: Deep Reinforcement Learning with Information Structure and PSR 🧠

This project investigates the hypothesis that incorporating **Information Structure (IS)** and **Predictive State Representations (PSR)** as an explicit **inductive bias** can significantly improve the efficiency and stability of agents in **Partially Observable Markov Decision Processes (POMDPs)**.

The work implements an **IS-DQN** agent and evaluates it against strong modern baselines across three environments with different forms of partial observability.

---

## 1. Theoretical Foundation: Information Structure and PSR

### 1.1 The Information Structure (IS) Framework

According to the paper *“On the Role of Information Structure in Reinforcement Learning for Partially-Observable Sequential Teams and Games”*, **Information Structure** describes the causal dependencies among all variables in the system (states, actions, observations).

* **Information-Structural State ($\mathbb{I}_{h}^{\dagger}$):**  
  A key contribution of the paper is the definition of the minimal set of past variables (latent or observed) required to predict future observations.

* **PSR and Dynamical Rank:**  
  Predictive State Representation theory shows that the dynamical complexity of the environment (its “dynamical rank”) is directly bounded by the size of the structural-informational state.  
  IS-DQN leverages this fact to construct a compact and highly informative state embedding.

---

## 2. Project Scope and Experimental Methodology

This study evaluates IS-DQN in three challenging POMDP environments:

| Environment | Description | Observability Challenge |
| :--- | :--- | :--- |
| **TwoTigers** | A classic POMDP requiring belief reasoning and memory to locate the tiger. | Bayesian belief updates & exploration |
| **KeyDoorMazeEnv** | A maze navigation environment with a key and a door. | Long-term memory & dependency on a key event |
| **DelayedStochasticObsEnv** | The agent receives observations with stochastic delay. | State inference with delayed/noisy information |

### 2.1 Strong Baselines

IS-DQN is compared against well-established architectures:

* **Vanilla DQN:** Uses only the raw observation or oracle belief.
* **DRQN:** A recurrent DQN with LSTM/GRU to maintain internal memory over sequences.
* **PPO-LSTM:** A strong policy-gradient baseline capable of handling partial observability.

---

## 3. Implementations of $\text{IS-DQN}$

The core IS-DQN architecture combines oracle belief with either a **spectral PSR embedding** or a **learned PSR embedding** as part of the DQN input.

| IS-DQN Variant | Reference File | Description |
| :--- | :--- | :--- |
| **Spectral IS-DQN** | `IS_tigers_3.py` | Uses SVD over the History × Test matrix to obtain a minimally sufficient predictive embedding. |
| **Learned IS-DQN** | `IS_tigers_2.py` | Learns PSR tests online with a neural network, providing predictive features to the DQN. |

---

## 4. Ablation Study and Algorithmic Modifications

Ablation experiments were conducted to isolate the contribution of sampling and exploration mechanisms:

| Ablation Setting | Modification | Purpose |
| :--- | :--- | :--- |
| **IS-DQN (Full)** | Uses PER and entropy-based exploration | Evaluate best-case performance |
| **IS-DQN w/o PER** | Uniform replay sampling | Assess importance of prioritization |
| **IS-DQN w/o ENTROPY** | Only ε-greedy exploration | Evaluate intrinsic/exploration bonuses |
| **IS-DQN Minimal** | No PER, no entropy, only PSR input | Measure the pure contribution of the PSR embedding |

---

## 5. Future Work 🚀

1. **Formal Estimation of $\mathbb{I}_{h}^{\dagger}$:**  
   Approximate the information-structural state size and correlate it with the optimal PSR dimension.

2. **IS-DRQN Architecture:**  
   Integrate PSR embeddings into a recurrent agent, using PSR features to modulate or initialize the hidden state.

3. **State Space Visualization:**  
   Use t-SNE or UMAP to compare how PSR embeddings cluster critical histories vs. the belief space.

