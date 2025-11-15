from Entornos.TwoTigersEnv import TwoTigersEnv
from .DQN import DQNAgent

import numpy as np
import random
from tqdm import trange
import torch
from torch.utils.tensorboard import SummaryWriter


def update_psr(p_t, a_t, o_t1):
    """
    PSR update for a single tiger.
    
    Args:
        p_t: array([pL, pR]) - current beliefs
        a_t: str - action taken: "AL", "AR", "AE"
        o_t1: str - observation received: "OL", "OR"
    """
    if a_t == "AE":  # Listen
        if o_t1 == "OL":
            numerator_L = 0.85 * p_t[0]
            numerator_R = 0.15 * p_t[1]
        else:  # o_t1 == "OR"
            numerator_L = 0.15 * p_t[0]
            numerator_R = 0.85 * p_t[1]
        total = numerator_L + numerator_R
        p_next = np.array([numerator_L / total, numerator_R / total])
    else:  # Open door
        p_next = np.array([0.5, 0.5])
    return p_next


def check_gpu():
    """Check if PyTorch can use GPU."""
    print(f"\n{'='*40}")
    print(" Checking GPU availability (PyTorch)...")
    print(f"{'='*40}")
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        current_dev_idx = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_dev_idx)
        print(f"✅ GPU available.")
        print(f"   Devices found: {device_count}")
        print(f"   Using device {current_dev_idx}: {device_name}")
        return torch.device("cuda")
    else:
        print(f"❌ GPU not available.")
        print("   Training will run on CPU.")
        return torch.device("cpu")


ACTION_MAP = {
    0: ("AL", "AL"), 1: ("AL", "AR"), 2: ("AL", "AE"),
    3: ("AR", "AL"), 4: ("AR", "AR"), 5: ("AR", "AE"),
    6: ("AE", "AL"), 7: ("AE", "AR"), 8: ("AE", "AE")
}


def main():
    # Environment setup
    env = TwoTigersEnv(max_episode_steps=1000)
    action_space = env.action_space_()
    state_space = env.state_space_()

    # Hyperparameters
    gamma = 0.99
    batch_size = 64
    lr = 1e-2
    episodes = 1000
    
    # Device
    device = check_gpu()

    # Model
    model = DQNAgent(state_space, action_space, device=device, lr=lr, gamma=gamma, batch_size=batch_size)

    # TensorBoard writer
    writer = SummaryWriter()

    # Epsilon decay schedule
    eps_start = 0.5
    eps_end = 0.01
    eps_decay = 0.995

    for episode in trange(episodes):
        epsilon = max(eps_end, eps_start * (eps_decay ** episode))
        
        env.reset()
        p_t = np.array([0.5, 0.5, 0.5, 0.5])
        
        total_reward = 0
        steps_in_episode = 0
        losses = []
        done = False
        
        while not done:
            # Select action based on PSR state
            action_int = model.q_net.sample_action(torch.FloatTensor(p_t).to(device), epsilon=epsilon)
            action_str_tuple = ACTION_MAP[action_int]

            # Step environment
            next_obs_tuple, reward, done, truncated, info = env.step(action_int)
            episode_done = done or truncated

            # Update PSR state
            p_t_tiger1 = p_t[0:2]
            p_t_tiger2 = p_t[2:4]
            
            p_t_next_tiger1 = update_psr(p_t_tiger1, action_str_tuple[0], next_obs_tuple[0])
            p_t_next_tiger2 = update_psr(p_t_tiger2, action_str_tuple[1], next_obs_tuple[1])
            
            p_t_next = np.concatenate([p_t_next_tiger1, p_t_next_tiger2])
            
            # Validate PSR state
            if np.any(np.isnan(p_t_next)) or np.any(np.isinf(p_t_next)):
                print(f"⚠️  WARNING: PSR state contains NaN/Inf!")
                p_t_next = np.array([0.5, 0.5, 0.5, 0.5])

            # Store transition
            model.replay_buffer.put(p_t, action_int, reward, p_t_next, episode_done)
            
            p_t = p_t_next
            total_reward += reward
            steps_in_episode += 1

            # Train model
            if len(model.replay_buffer) >= batch_size:
                loss = model.update()
                if loss is not None:
                    losses.append(loss)
                if steps_in_episode % 5 == 0:
                    model.soft_update(tau=1e-2)
            
            # Log progress
            if steps_in_episode % 10 == 0:
                avg_loss = np.mean(losses[-10:]) if losses else 0
                print(f"  Episode {episode+1}, Step {steps_in_episode}, Buffer: {len(model.replay_buffer)}, Loss: {avg_loss:.4f}, Reward: {total_reward}")
            
            if episode_done:
                break
        
        # End of episode logging
        avg_episode_loss = np.mean(losses) if losses else 0
        print(f"Episode {episode+1} DONE - Reward: {total_reward}, Steps: {steps_in_episode}, Buffer: {len(model.replay_buffer)}, Loss: {avg_episode_loss:.4f}")
        
        writer.add_scalar("Reward/episode", total_reward, episode)
        writer.add_scalar("Steps/episode", steps_in_episode, episode)
        writer.add_scalar("Loss/episode", avg_episode_loss, episode)

    writer.close()


if __name__ == "__main__":
    main()
