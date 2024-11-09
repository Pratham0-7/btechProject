import gym
from gym import spaces
import numpy as np
import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig

# Step 1: Define the Custom Environment
class MyEnv(gym.Env):
    def __init__(self, config=None):
        super(MyEnv, self).__init__()

        # Define action space (Discrete or Continuous)
        self.action_space = spaces.Discrete(2)  # Example with 2 discrete actions

        # Define observation space (could be discrete or continuous)
        self.observation_space = spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)

    def reset(self):
        # Initialize state
        self.state = np.random.rand(4)  # Random initial state
        return self.state

    def step(self, action):
        # Define reward, next_state, done based on action
        reward = 1 if action == 1 else 0  # Reward based on the action
        self.state = np.random.rand(4)  # Generate a new random state
        done = False  # Set to True if episode ends
        info = {}

        return self.state, reward, done, info

# Step 2: Register the Environment
def env_creator(config):
    return MyEnv(config)

register_env("my_custom_env", env_creator)

# Step 3: Set up PPO with Ray RLlib
# Initialize Ray
ray.init(ignore_reinit_error=True)

# Define PPOConfig
ppo_config = (
    PPOConfig()
    .environment("my_custom_env", env_config={})  # Register the custom environment
    .framework("torch")  # Use torch or tf
    .rollouts(num_rollout_workers=1)  # Adjust workers if needed
)

# Build the PPO trainer
ppo = ppo_config.build()

# Step 4: Train for a number of iterations
for i in range(10):
    result = ppo.train()
    print(f"Iteration: {i}, reward: {result['episode_reward_mean']}")

# Clean up Ray
ray.shutdown()
