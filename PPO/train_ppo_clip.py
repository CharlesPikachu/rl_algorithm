'''
Function:
    PPO-CLIP algorithm
Author:
    Zhenchao Jin
'''
import gym
import torch
import imageio
import numpy as np
import torch.nn as nn
from torch.optim import Adam
import matplotlib.pyplot as plt
import torch.nn.functional as F


'''hyperparameters'''
EPISODES = 500
GAMMA = 0.99
LAMBDA = 0.95
CLIP_EPS = 0.2
LR = 3e-4
BATCH_SIZE = 2048
EPOCHS = 5


'''ActorCritic'''
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(), nn.Linear(128, action_dim), nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(), nn.Linear(128, 1)
        )
    '''forward'''
    def forward(self, state):
        probs = self.actor(state)
        value = self.critic(state)
        return probs, value


'''advantage estimate'''
def compute_advantage(rewards, values, dones, gamma=GAMMA, lam=LAMBDA):
    advantages, gae, next_value = [], 0, 0
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * next_value * (1 - dones[i]) - values[i]
        gae = delta + gamma * lam * (1 - dones[i]) * gae
        advantages.insert(0, gae)
        next_value = values[i]
    return torch.tensor(advantages, dtype=torch.float32)


'''video recorder'''
class Recorder:
    def __init__(self, env, path):
        self.env = env
        self.frames = []
        self.path = path
    '''record'''
    def record(self):
        self.frames.append(self.env.render())
    '''save'''
    def save(self):
        imageio.mimsave(self.path, self.frames, fps=30)


'''PPOClip'''
class PPOClip:
    def __init__(self, state_dim, action_dim):
        self.model = ActorCritic(state_dim, action_dim)
        self.optimizer = Adam(self.model.parameters(), lr=LR)
    '''update'''
    def update(self, trajectories):
        states = torch.stack(trajectories['states'])
        actions = torch.tensor(trajectories['actions'])
        old_log_probs = torch.stack(trajectories['log_probs']).detach()
        rewards = trajectories['rewards']
        dones = trajectories['dones']
        values = torch.stack(trajectories['values']).detach().squeeze()
        advantages = compute_advantage(rewards, values, dones)
        returns = advantages + values
        for _ in range(EPOCHS):
            probs, value_est = self.model(states)
            dist = torch.distributions.Categorical(probs)
            log_probs = dist.log_prob(actions)
            ratio = torch.exp(log_probs - old_log_probs)
            clip_adv = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * advantages
            loss_policy = -torch.min(ratio * advantages, clip_adv).mean()
            loss_value = F.mse_loss(value_est.squeeze(), returns)
            loss = loss_policy + 0.5 * loss_value
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


'''train'''
def train():
    env = gym.make("LunarLander-v2", render_mode="rgb_array")
    agent = PPOClip(env.observation_space.shape[0], env.action_space.n)
    episode_rewards = []
    recorder = Recorder(env, "ppo_clip_lander.mp4")
    state_buffer, action_buffer, logprob_buffer, value_buffer, reward_buffer, done_buffer = [], [], [], [], [], []
    state, _ = env.reset()
    ep_reward = 0
    total_steps = 0
    for episode in range(EPISODES):
        state, _ = env.reset()
        done = False
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            probs, value = agent.model(state_tensor)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            state_buffer.append(state_tensor.squeeze())
            action_buffer.append(action.item())
            logprob_buffer.append(log_prob)
            value_buffer.append(value.squeeze())
            reward_buffer.append(reward)
            done_buffer.append(done)
            state = next_state
            ep_reward += reward
            total_steps += 1
            if episode >= EPISODES - 5:
                recorder.record()
            if total_steps % BATCH_SIZE == 0:
                agent.update({
                    'states': state_buffer, 'actions': action_buffer, 'log_probs': logprob_buffer, 'rewards': reward_buffer, 'dones': done_buffer, 'values': value_buffer,
                })
                state_buffer, action_buffer, logprob_buffer, value_buffer = [], [], [], []
                reward_buffer, done_buffer = [], []
        episode_rewards.append(ep_reward)
        print(f"Episode {episode}, Reward: {ep_reward}")
        ep_reward = 0
    recorder.save()
    env.close()
    plt.plot(episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("PPO-Clip on LunarLander-v2")
    plt.savefig("reward_curve.png")
    plt.close()
    np.save("rewards_clip.npy", episode_rewards)


'''debug'''
if __name__ == "__main__":
    train()