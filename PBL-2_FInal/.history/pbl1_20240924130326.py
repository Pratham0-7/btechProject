import gym
from gym import spaces
import numpy as np
import traci
from ray import tune
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import torch
import torch.nn as nn
import ray

# Step 1: Define the Custom SUMO Environment
class SumoEnv(gym.Env):
    def __init__(self, config):
        super(SumoEnv, self).__init__()

        # Define the action space (5 discrete actions)
        self.action_space = spaces.Discrete(5)
        
        # Define the observation space (10 features for example)
        self.observation_space = spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)

        # Initialize SUMO environment
        self._initialize_sumo()

    def _initialize_sumo(self):
        sumoCmd = ["sumo", "-c", "D:\\PBL-2_FInal\\map.sumocfg"]  # Update with your SUMO config path
        traci.start(sumoCmd)

    def reset(self):
        traci.close()  # Close any existing connection
        self._initialize_sumo()  # Restart the SUMO process
        return self._get_initial_observation()

    def step(self, action):
        self._apply_action(action)
        traci.simulationStep()
        observation = self._get_observation()
        reward = self._calculate_reward()
        done = self._check_done()
        info = {}
        return observation, reward, done, info

    def _apply_action(self, action):
        # Example action mapping
        pass

    def _get_observation(self):
        # Replace with actual state retrieval logic
        state_data = np.random.rand(10)  # Placeholder for actual state
        return state_data

    def _calculate_reward(self):
        return 1.0  # Placeholder for actual reward calculation

    def _check_done(self):
        return traci.simulation.getMinExpectedNumber() <= 0  # Modify as needed

    def _get_initial_observation(self):
        return self._get_observation()

    def render(self, mode='human'):
        pass

    def close(self):
        traci.close()


# Step 2: Define the Custom Model (if needed)
class CustomTorchModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.fc1 = nn.Linear(obs_space.shape[0], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_out = nn.Linear(256, num_outputs)

    def forward(self, input_dict, state, seq_lens):
        x = torch.relu(self.fc1(input_dict["obs"]))
        x = torch.relu(self.fc2(x))
        return self.fc_out(x), state

    def value_function(self):
        return torch.zeros(1)  # Update if necessary


# Register the custom model
ModelCatalog.register_custom_model("my_model", CustomTorchModel)

# Initialize Ray
ray.init()

# Configure the PPO trainer
config = (
    PPOConfig()
    .environment(SumoEnv)  # Use the new method to set the environment
    .framework("torch")     # Use PyTorch as the framework
    .rollouts(num_rollout_workers=2)  # Set number of workers
    .training(
        model={
            "custom_model": "my_model",  # Specify the custom model
            "fcnet_hiddens": [256, 256],  # Hidden layer sizes
            "fcnet_activation": "relu"     # Activation function
        }
    )
    .resources(num_gpus=1)  # Set number of GPUs if available
)

# Create the PPO trainer
trainer = config.build()

# Training loop
for i in range(100):
    result = trainer.train()
    print(f"Iteration {i}: reward = {result['episode_reward_mean']}")

# Optionally save the model
trainer.save("path/to/save/model")

# Shutdown Ray
ray.shutdown()
