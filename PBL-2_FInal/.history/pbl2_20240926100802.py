import gym
from gym import spaces
import numpy as np
import ray
from ray import rllib
from ray.rllib.agents.ppo import PPO
from ray.tune import register_env

# Custom Environment
class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space and observation space as the same
        self.action_space = spaces.Discrete(4)  # Example: 4 discrete actions
        self.observation_space = spaces.Discrete(4)  # Same as action space

        # Initialize state (for example, a simple state)
        self.state = 0

    def reset(self):
        self.state = 0  # Reset state
        return self.state  # Return initial observation

    def step(self, action):
        # Simple environment logic (customize as needed)
        if action == 0:  # Action 0
            self.state += 1
        elif action == 1:  # Action 1
            self.state -= 1
        # Add more action logic as needed

        # Define reward and done conditions
        reward = 1 if self.state == 10 else -1  # Example reward logic
        done = True if self.state >= 10 else False  # Example done condition

        return self.state, reward, done, {}

    def render(self, mode='human'):
        pass  # Implement rendering if needed

    def close(self):
        pass  # Cleanup if needed

# Initialize Ray
ray.init()

# Register the custom environment
def env_creator(_):
    return CustomEnv()  # Return an instance of your custom environment

register_env("CustomEnv", env_creator)

# Configure the PPO trainer
config = {
    "env": "CustomEnv",
    "num_workers": 1,  # Adjust based on your setup
    "framework": "torch",  # Or "tf" for TensorFlow
    "multiagent": {
        "policies": {
            "policy_1": (None, CustomEnv().observation_space, CustomEnv().action_space, {}),
        },
        "policy_mapping_fn": (lambda agent_id: "policy_1"),
    },
}

# Create the PPO Trainer
trainer = PPO(config=config)

# Training loop
for i in range(100):  # Adjust number of iterations as needed
    result = trainer.train()
    print(f"Iteration {i}: {result['episode_reward_mean']}")

# Shutdown Ray after training
ray.shutdown()
