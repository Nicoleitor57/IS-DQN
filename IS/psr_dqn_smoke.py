"""Smoke runner: run one episode of the PSR+DQN variant (uses classes in IS_tigers_2.py)
This won't run the full training loop and is safe to execute while a long experiment runs.
"""
from IS.IS_tigers_2 import PSRPredictor, PSRBuffer, build_input_features
from Entornos.TwoTigersEnv import TwoTigersEnv
from IS.DQN import DQNAgent

import numpy as np
import torch


def check_gpu():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def run_one_episode():
    device = check_gpu()
    env = TwoTigersEnv(max_episode_steps=100)
    action_space = env.action_space_()
    # instantiate small PSR predictor and DQN like in IS_tigers_2
    psr = PSRPredictor(19)
    psr.to(device)
    dqn = DQNAgent(6, action_space, device=device, lr=1e-3, gamma=0.99, batch_size=16)

    obs, info = env.reset()
    p_t = np.array([0.5,0.5,0.5,0.5], dtype=np.float32)
    last_a = None
    last_o = None

    total_reward = 0
    steps = 0

    while True:
        psr_in = build_input_features(p_t, last_a, last_o)
        with torch.no_grad():
            psr_pred = psr(torch.FloatTensor(psr_in).unsqueeze(0).to(device)).cpu().numpy().reshape(-1)
        state_enh = np.concatenate([p_t, psr_pred]).astype(np.float32)
        action_int = dqn.q_net.sample_action(torch.FloatTensor(state_enh).to(device), epsilon=0.5)
        next_obs, reward, done, truncated, info = env.step(action_int)
        episode_done = done or truncated

        # simple PSR update like IS_tigers_2
        act1 = action_int // 3
        act2 = action_int % 3
        obs_map = {0: None, 1: 'OL', 2: 'OR'}
        a_tuple = ("AL","AR","AE")[act1], ("AL","AR","AE")[act2]
        o_tuple = (obs_map[next_obs[0]], obs_map[next_obs[1]])

        # update beliefs
        p1 = p_t[0:2]; p2 = p_t[2:4]
        if a_tuple[0] == 'AE':
            if o_tuple[0] == 'OL':
                numL = 0.85*p1[0]; numR = 0.15*p1[1]
            else:
                numL = 0.15*p1[0]; numR = 0.85*p1[1]
            p1 = np.array([numL/(numL+numR), numR/(numL+numR)])
        else:
            p1 = np.array([0.5,0.5])
        if a_tuple[1] == 'AE':
            if o_tuple[1] == 'OL':
                numL = 0.85*p2[0]; numR = 0.15*p2[1]
            else:
                numL = 0.15*p2[0]; numR = 0.85*p2[1]
            p2 = np.array([numL/(numL+numR), numR/(numL+numR)])
        else:
            p2 = np.array([0.5,0.5])

        p_t = np.concatenate([p1,p2])
        total_reward += reward
        steps += 1
        last_a = action_int
        last_o = (int(next_obs[0]), int(next_obs[1]))

        if episode_done or steps >= 100:
            break

    print(f"Smoke run finished: reward={total_reward}, steps={steps}")

if __name__ == '__main__':
    run_one_episode()
