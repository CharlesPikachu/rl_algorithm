'''
Function:
    Reinforce algorithm with / without baseline
Author:
    Zhenchao Jin
'''
import gym
import torch
import imageio
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.distributions import Categorical


'''save_video_mp4'''
def save_video_mp4(frames, path="out.mp4", fps=30):
    writer = imageio.get_writer(path, fps=fps, codec='libx264')
    for frame in tqdm(frames):
        writer.append_data(frame)
    writer.close()


'''PolicyNet'''
class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(), nn.Linear(128, action_dim), nn.Softmax(dim=-1)
        )
    '''forward'''
    def forward(self, x):
        return self.net(x)


'''ValueNet'''
class ValueNet(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(), nn.Linear(128, 1)
        )
    '''forward'''
    def forward(self, x):
        return self.net(x).squeeze(-1)


'''train_reinforce'''
def train_reinforce(env_name="LunarLander-v2", episodes=300, use_baseline=False, gif_path="out.gif", reward_curve_path="reward.png"):
    # init env
    env = gym.make(env_name, render_mode="rgb_array")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    # init policy and value net
    policy = PolicyNet(state_dim, action_dim)
    value_net = ValueNet(state_dim) if use_baseline else None
    # init optimizer
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)
    value_optimizer = optim.Adam(value_net.parameters(), lr=1e-3) if use_baseline else None
    # init hyper-parameters
    gamma = 0.99
    # start
    frames = []
    reward_history = []
    for episode in range(episodes):
        state, _ = env.reset()
        log_probs, rewards, states = [], [], []
        total_reward = 0
        done = False
        step = 0
        while not done:
            if step % 5 == 0:
                frames.append(env.render())
            state_tensor = torch.tensor(state, dtype=torch.float32)
            probs = policy(state_tensor)
            dist = Categorical(probs)
            action = dist.sample()
            log_probs.append(dist.log_prob(action))
            states.append(state_tensor)
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            rewards.append(reward)
            total_reward += reward
            state = next_state
            step += 1
        reward_history.append(total_reward)
        # compute returns
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8 + 1e-6)
        # whether to use baseline
        if use_baseline:
            states_tensor = torch.stack(states)
            values = value_net(states_tensor).detach()
            advantages = returns - values
            loss = -torch.stack([lp * adv for lp, adv in zip(log_probs, advantages)]).sum()
            value_loss = nn.functional.mse_loss(value_net(states_tensor), returns)
            value_optimizer.zero_grad()
            value_loss.backward()
            value_optimizer.step()
        else:
            loss = -torch.stack([lp * R for lp, R in zip(log_probs, returns)]).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"[{episode+1}/{episodes}] Reward: {total_reward:.1f}")
    env.close()
    # save results
    save_video_mp4(frames, gif_path, fps=30)
    print(f"Saved animation to {gif_path}")
    plt.plot(reward_history)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("REINFORCE with{}".format(" Baseline" if use_baseline else "out Baseline"))
    plt.grid(True)
    plt.savefig(reward_curve_path)
    print(f"Saved reward curve to {reward_curve_path}")
    return reward_history


'''run'''
if __name__ == '__main__':
    r1 = train_reinforce(use_baseline=False, gif_path="nobaseline.mp4", reward_curve_path="nobaseline.png")
    r2 = train_reinforce(use_baseline=True, gif_path="baseline.mp4", reward_curve_path="baseline.png")
    plt.plot(r1, label="Without Baseline")
    plt.plot(r2, label="With Baseline")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("REINFORCE Comparison")
    plt.legend()
    plt.grid(True)
    plt.savefig("reward_curve_compare.png")
    print("Saved comparison to reward_curve_compare.png")