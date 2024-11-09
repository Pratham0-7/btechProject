import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Simulate common data
episodes = np.arange(0, 101)
long_episodes = np.arange(0, 201)

# 1. Comparison of PPO and SAC Efficiency Over Time (Figure 10)
ppo_time = 0.8 + 0.2 * np.random.rand(len(episodes))  # PPO time per episode
sac_time = 1.0 + 0.1 * np.random.rand(len(episodes))  # SAC time per episode

plt.figure(figsize=(10, 6))
plt.plot(episodes, ppo_time, color='blue', label='PPO')
plt.plot(episodes, sac_time, color='green', label='SAC')
plt.xlabel('Episodes')
plt.ylabel('Time per Episode (seconds)')
plt.title('Comparison of PPO and SAC Efficiency Over Time')
plt.legend()
plt.show()

# 2. Rewards Per Episode for PPO and SAC (Figure 11)
ppo_rewards = 0.9 + 0.2 * np.random.rand(len(episodes))
sac_rewards = 1.0 + 0.2 * np.random.rand(len(episodes))

plt.figure(figsize=(10, 6))
plt.plot(episodes, ppo_rewards, color='blue', linestyle='-', label='PPO')
plt.plot(episodes, sac_rewards, color='red', linestyle='--', label='SAC')
plt.xlabel('Episodes')
plt.ylabel('Reward per Episode')
plt.title('Rewards per Episode for PPO and SAC')
plt.legend()
plt.show()

# 3. Battery Levels vs Distance to Charging Station (Figure 12)
distance = np.linspace(0, 100, 100)
ppo_battery = np.clip(100 - distance + np.random.normal(0, 10, 100), 0, 100)
sac_battery = np.clip(100 - distance + np.random.normal(0, 10, 100), 0, 100)

plt.figure(figsize=(10, 6))
plt.scatter(distance, ppo_battery, color='blue', label='PPO Battery Levels')
plt.scatter(distance, sac_battery, color='red', label='SAC Battery Levels')
plt.xlabel('Distance to Charging Station')
plt.ylabel('Battery Level')
plt.title('Battery Levels vs Distance to Charging Station')
plt.legend()
plt.show()

# 4. Comparison of PPO and SAC Actor Loss Over Time (Figure 13)
ppo_loss = 0.5 + 0.2 * np.random.rand(len(episodes))
sac_loss = 0.5 + 0.2 * np.random.rand(len(episodes))

plt.figure(figsize=(10, 6))
plt.plot(episodes, ppo_loss, color='blue', label='PPO Actor Loss')
plt.plot(episodes, sac_loss, color='green', label='SAC Actor Loss')
plt.xlabel('Episodes')
plt.ylabel('Actor Loss')
plt.title('Comparison of PPO and SAC Actor Loss Over Time')
plt.legend()
plt.show()

# 5. Loss vs. Episodes (Final Figure)
ppo_loss_long = 0.6 - 0.003 * long_episodes + 0.02 * np.random.rand(len(long_episodes))
sac_loss_long = 0.55 - 0.003 * long_episodes + 0.02 * np.random.rand(len(long_episodes))

plt.figure(figsize=(10, 6))
plt.plot(long_episodes, ppo_loss_long, color='blue', label='PPO')
plt.plot(long_episodes, sac_loss_long, color='yellow', label='SAC')
plt.xlabel('Episodes')
plt.ylabel('Smoothed Loss')
plt.title('Loss vs. Episodes')
plt.legend()
plt.show()
