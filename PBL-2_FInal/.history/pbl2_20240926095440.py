import gym
from gym import spaces
import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.env import MultiAgentEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import torch
import torch.nn as nn

# Step 1: Define the Custom Environment
class MyCustomEnv(MultiAgentEnv):
    def __init__(self, config):
        super().__init__()
        self.action_space = spaces.Discrete(2)  # Two possible actions
        self.observation_space = spaces.Box(low=0, high=1, shape=(4,), dtype=float)  # Example observation space
        self.state = None

    def reset(self):
        self.state = self.observation_space.sample()  # Sample a random state
        return {0: self.state}  # Return initial state for agent 0

    def step(self, action):
        # Implement your environment's logic here
        reward = 1.0 if action == 1 else -1.0  # Example reward logic
        self.state = self.observation_space.sample()  # Sample new state
        done = False  # Define a condition for when the episode is done
        return {0: self.state}, reward, done, {}  # Return state, reward, done, and info

# Step 2: Define the Custom Torch Model
class CustomTorchModel(TorchModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        self.fc = nn.Sequential(
            nn.Linear(obs_space.shape[0], 64),
            nn.ReLU(),
            nn.Linear(64, num_outputs)
        )

    def forward(self, input_dict, state, seq_lens):
        x = input_dict["obs"]
        x = self.fc(x)
        return x, state

    def value_function(self):
        return self.fc[-1].weight  # Example, modify as per your logic

# Step 3: Register the Environment and Custom Model
ray.init(ignore_reinit_error=True)
tune.register_env("my_custom_env", lambda config: MyCustomEnv(config))
ModelCatalog.register_custom_model("custom_torch_model", CustomTorchModel)

# Step 4: Configure and Train the PPO Agent
config = {
    "env": "my_custom_env",
    "num_workers": 1,
    "framework": "torch",
    "model": {
        'custom_model': "custom_torch_model",
    },
}

trainer = PPOTrainer(config=config)

# Step 5: Run the Training Loop
for i in range(10):  # Number of training iterations
    result = trainer.train()
    print(f"Iteration {i}: reward {result['episode_reward_mean']}")

# Cleanup Ray
ray.shutdown()
