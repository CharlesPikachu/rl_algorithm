import numpy as np
import matplotlib.pyplot as plt

clip = np.load("rewards_clip.npy")
kl = np.load("rewards_klpenalty.npy")
vanilla = np.load("rewards_vanilla.npy")

plt.figure(figsize=(10, 6))
plt.plot(clip, label="PPO-Clip", color="blue")
plt.plot(kl, label="PPO-KL Penalty", color="green")
plt.plot(vanilla, label="Vanilla PPO", color="red")

plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Reward Comparison Across PPO Variants")
plt.legend()
plt.grid(True)
plt.savefig("ppo_comparison.png")
plt.close()