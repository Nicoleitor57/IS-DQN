# PSR + DQN Variants for TwoTigers Environment

This directory contains three implementations of DQN agents for the Two-Tigers POMDP, with increasing levels of PSR (Predictive State Representation) inductive bias. All variants use oracle Bayesian belief updates for the tiger positions and differ in how they represent history/predictive features.

## Variants

### 1. `IS_tigers.py` - Oracle Belief + DQN (Baseline)
**Approach:** 
- State: 4-dim belief vector (per-tiger probabilities p_left, p_right).
- Features: Direct Bayesian beliefs only (no predictive tests).
- DQN: Vanilla DQN with replay buffer and soft target updates.

**Use case:** Quick baseline; validates training loop and environment integration.

**Run:**
```bash
python -m IS.IS_tigers
```

---

### 2. `IS_tigers_2.py` - Learned Predictive Tests + DQN
**Approach:**
- State: 4-dim belief vector + 2-dim learned PSR predictions.
- PSR module: Small neural network trained online to predict P(OL | listen) for each tiger.
- Training: Supervised learning on (history, test outcome) pairs whenever a listen action occurs.
- Features are concatenated: belief (4) + PSR predictions (2) → 6-dim DQN input.

**Inductive bias:** Predictions of near-future test outcomes help disambiguate history.

**Paper connection:** Simple approximation of test-based PSR; learns what tests are predictive.

**Run:**
```bash
python -m IS.IS_tigers_2
# Quick smoke test (1 episode):
python -m IS.psr_dqn_smoke
```

---

### 3. `IS_tigers_3.py` - Spectral PSR + DQN
**Approach:**
- Phase 1 (Warmup): Collect N transitions (default 1000) with random policy.
  - Record (history, test_outcome) pairs where:
    - History: recent (action, observation) tuple.
    - Tests: P(OL | listen on tiger i) for i=1,2 → 2 tests.
- Phase 2 (SVD): Build empirical matrix P_HT (|H| × |T|), compute SVD, extract top-r singular vectors.
  - P_HT[h, t] = empirical probability of test outcome t given history h.
  - U_r: projection matrix (|H| × r) forming r-dim PSR embeddings.
- Phase 3 (Training): At runtime, map history → embedding via U_r, concatenate with belief for DQN.

**Features:** belief (4) + spectral PSR embedding (r=2, default) → 6-dim DQN input.

**Inductive bias:** Low-rank structure of history space; captures essential predictive information.

**Paper connection:** Direct implementation of spectral PSR ideas (Boots et al. 2013, Holmes et al. 2010). 
- Histories → Tests matrix captures core PSR theory.
- SVD extracts minimal sufficient statistics for future predictions.
- Can be compared to oracle PSR (if true rank is known) to validate dimensionality.

**Run:**
```bash
# Full training (200 episodes, 1000-step warmup):
python -m IS.IS_tigers_3

# Quick smoke test (warmup=200, episode=1):
python -m IS.spectral_psr_smoke
```

---

## Comparison & Paper Alignment

| Aspect | IS_tigers | IS_tigers_2 | IS_tigers_3 |
|--------|-----------|-------------|------------|
| **Belief state** | Bayesian oracle | Bayesian oracle | Bayesian oracle |
| **History features** | None | Online learned neural net | Spectral (SVD-based) |
| **Test definition** | N/A | P(OL\|listen) per tiger | P(OL\|listen) per tiger |
| **Learning** | N/A | Supervised (online) | Spectral decomposition (batch) |
| **DQN input dim** | 4 | 6 | 6 (default r=2) |
| **Paper alignment** | Baseline | Simple test predictor | Full spectral PSR theory |

---

## Hyperparameters

All scripts use:
- `gamma=0.99` (discount factor)
- `batch_size=64`
- `lr=1e-2` (DQN learning rate)
- `episodes=1000` (or configurable)
- `max_episode_steps=1000`
- Epsilon decay: 0.5 → 0.01 over 0.995^episode

For `IS_tigers_3.py` specifically:
- `warmup_transitions=1000`: transitions to collect before SVD.
- `rank=2`: dimensionality of spectral PSR embedding.

---

## TensorBoard Logging

All variants log to TensorBoard (default: `runs/` directory).

View results:
```bash
tensorboard --logdir=runs/
```

Metrics:
- `Reward/episode`: Episode cumulative reward.
- `Loss/episode`: Average DQN training loss per episode.
- `Steps/episode`: Episode length (should be ~1000 for healthy learning).

---

## Next Steps / Extensions

1. **Multiple seeds:** Run each variant with 5–10 seeds, aggregate mean ± std or IC.
2. **Learned PSR via EM:** Extend `IS_tigers_2` with EM-style parameter learning for more principled test prediction.
3. **Recurrent architectures:** Use GRU/LSTM in place of feedforward DQN for richer history representation.
4. **Double DQN / Prioritized Replay:** Add RL improvements (especially for POMDP stability).
5. **Paper comparison:** If your paper specifies exact spectral algorithm (e.g., spectral methods with covariates, model-based PSR), map this implementation to those sections.

---

## Debugging / Validation

- **Episode termination:** All scripts check `episode_done = done or truncated` per Gymnasium spec.
- **Belief validation:** Beliefs checked for NaN/Inf; auto-reset to [0.5, 0.5] on corruption.
- **Smoke tests:** Use `psr_dqn_smoke.py` and `spectral_psr_smoke.py` to validate without long training.

