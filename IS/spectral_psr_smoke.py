"""Smoke runner for spectral PSR + DQN variant (IS_tigers_3.py)
Tests warmup phase (collecting histories) and one training episode.
Safe to run in parallel with long training.
"""
from IS.IS_tigers_3 import SpectralPSR
from Entornos.TwoTigersEnv import TwoTigersEnv

import numpy as np


def test_spectral_psr():
    print("Testing SpectralPSR...")
    psr = SpectralPSR(rank=2, max_histories=100)
    
    # Simulate collecting a few history-test pairs
    for i in range(50):
        history = (i % 5, (i % 2, i % 3))
        test_outcome = (i % 2, (i + 1) % 2)
        psr.add_observation(history, test_outcome)
    
    # Fit
    rank = psr.fit()
    print(f"  PSR fitted with effective rank: {rank}")
    
    # Test embedding
    emb = psr.embed_history((0, (0, 0)))
    print(f"  Sample embedding shape: {emb.shape}, values: {emb}")
    print("✅ SpectralPSR basic test passed.\n")


def test_warmup_and_one_episode():
    print("Testing warmup collection + one training episode...")
    env = TwoTigersEnv(max_episode_steps=50)
    psr = SpectralPSR(rank=2, max_histories=500)
    
    # Warmup: collect 200 transitions
    print("  Collecting 200 warmup transitions...")
    warmup_steps = 0
    while warmup_steps < 200:
        obs, info = env.reset()
        last_a = None
        last_o = None
        
        while warmup_steps < 200:
            history_key = (last_a, last_o)
            action_int = np.random.randint(0, 9)
            next_obs, reward, done, truncated, info = env.step(action_int)
            
            test_result = (
                1 if next_obs[0] == 1 else 0,
                1 if next_obs[1] == 1 else 0
            )
            psr.add_observation(history_key, test_result)
            
            last_a = action_int
            last_o = (int(next_obs[0]), int(next_obs[1]))
            warmup_steps += 1
            
            if done or truncated:
                break
    
    # Fit PSR
    rank = psr.fit()
    print(f"  PSR fitted with rank {rank}")
    
    # One episode with PSR
    print("  Running one training episode with PSR features...")
    obs, info = env.reset()
    p_t = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    last_a = None
    last_o = None
    
    total_reward = 0
    steps = 0
    
    while True:
        history_key = (last_a, last_o)
        psr_emb = psr.embed_history(history_key)
        state_enh = np.concatenate([p_t, psr_emb])
        
        action_int = np.random.randint(0, 9)
        next_obs, reward, done, truncated, info = env.step(action_int)
        
        total_reward += reward
        steps += 1
        last_a = action_int
        last_o = (int(next_obs[0]), int(next_obs[1]))
        
        if done or truncated or steps >= 50:
            break
    
    print(f"  Episode reward: {total_reward}, steps: {steps}")
    print("✅ Warmup + training episode test passed.\n")


if __name__ == '__main__':
    test_spectral_psr()
    test_warmup_and_one_episode()
    print("✅ All smoke tests passed!")
