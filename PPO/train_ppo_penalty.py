'''
Function:
    PPO-KL Penalty algorithm
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
LR = 3e-4
BATCH_SIZE = 2048
EPOCHS = 5
TARGET_KL = 0.01


'''ActorCritic'''
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU(), nn.Linear(128, action_dim), nn.Softmax(dim=-1))
        self.critic = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU(), nn.Linear(128, 1))
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


'''PPOKLPenalty'''
class PPOKLPenalty:
    def __init__(self, state_dim, action_dim):
        self.model = ActorCritic(state_dim, action_dim)
        self.optimizer = Adam(self.model.parameters(), lr=LR)
        self.beta = 1.0
    '''update'''
    def update(self, trajectories):
        states = torch.stack(trajectories['states'])
        actions = torch.tensor(trajectories['actions'])
        old_probs = torch.stack(trajectories['probs']).detach()
        rewards = trajectories['rewards']
        dones = trajectories['dones']
        values = torch.stack(trajectories['values']).detach().squeeze()
        advantages = compute_advantage(rewards, values, dones)
        returns = advantages + values
        for _ in range(EPOCHS):
            probs, values_pred = self.model(states)
            dist = torch.distributions.Categorical(probs)
            dist_old = torch.distributions.Categorical(old_probs)
            log_probs = dist.log_prob(actions)
            log_old = dist_old.log_prob(actions)
            ratio = torch.exp(log_probs - log_old)
            loss_policy = -(ratio * advantages).mean()
            loss_value = F.mse_loss(values_pred.squeeze(), returns)
            kl_div = torch.distributions.kl_divergence(dist_old, dist).mean()
            loss_total = loss_policy + 0.5 * loss_value + self.beta * kl_div
            self.optimizer.zero_grad()
            loss_total.backward()
            self.optimizer.step()
            if kl_div > 1.5 * TARGET_KL:
                self.beta *= 2
            elif kl_div < 0.5 * TARGET_KL:
                self.beta *= 0.5


'''train'''
def train():
    env = gym.make("LunarLander-v2", render_mode="rgb_array")
    agent = PPOKLPenalty(env.observation_space.shape[0], env.action_space.n)
    recorder = Recorder(env, "ppo_klpenalty_lander.mp4")
    episode_rewards = []
    state_buffer, action_buffer, prob_buffer, value_buffer, reward_buffer, done_buffer = [], [], [], [], [], []
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
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            state_buffer.append(state_tensor.squeeze())
            action_buffer.append(action.item())
            prob_buffer.append(probs.squeeze().detach())
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
                    'states': state_buffer, 'actions': action_buffer, 'probs': prob_buffer, 'rewards': reward_buffer, 'dones': done_buffer, 'values': value_buffer,
                })
                state_buffer, action_buffer, prob_buffer, value_buffer = [], [], [], []
                reward_buffer, done_buffer = [], []
        episode_rewards.append(ep_reward)
        print(f"Episode {episode}, Reward: {ep_reward}, Beta: {agent.beta:.3f}")
        ep_reward = 0
    recorder.save()
    env.close()
    plt.plot(episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("PPO with KL Penalty on LunarLander-v2")
    plt.savefig("reward_ppo_klpenalty.png")
    plt.close()
    np.save("rewards_klpenalty.npy", episode_rewards)


'''debug'''
if __name__ == "__main__":
    train()