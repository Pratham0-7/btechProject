import gym
from gym import spaces
import numpy as np
import traci  # This assumes SUMO's TraCI API is properly installed and accessible
from ray.rllib.env.env_context import EnvContext
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import torch
import torch.nn as nn
from ray import tune
from ray.rllib.algorithms.ppo import PPOTrainer  # Update the import statement

# Step 1: Define the Custom SUMO Environment

class SumoEnv(gym.Env):
    def __init__(self, config: EnvContext):
        super(SumoEnv, self).__init__()

        # Define the action space (e.g., 5 discrete actions)
        self.action_space = spaces.Discrete(5)
        
        # Define the observation space (adjust dimensions as needed)
        # Assuming 10 features for the observation, change based on your state representation
        self.observation_space = spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)

        # Initialize SUMO environment
        self._initialize_sumo()

    def _initialize_sumo(self):
        # Start SUMO with TraCI
        sumoBinary = "sumo"  # Use "sumo-gui" if you want visualization
        sumoCmd = [sumoBinary, "-c", "your_sumo_config.sumocfg"]  # Replace with your SUMO config path
        traci.start(sumoCmd)

    def reset(self):
        # Reset SUMO environment and get the initial observation
        traci.load(["-c", "your_sumo_config.sumocfg"])  # Reload the simulation
        return self._get_initial_observation()

    def step(self, action):
        # Apply the action to the environment (you need to implement how the action affects SUMO)
        self._apply_action(action)
        
        # Simulate one step in SUMO
        traci.simulationStep()

        # Get the new observation
        observation = self._get_observation()

        # Calculate reward (implement reward function based on your use case)
        reward = self._calculate_reward()

        # Determine if the episode is done (e.g., based on max steps or criteria)
        done = self._check_done()

        # Additional info (can be empty or contain debugging info)
        info = {}

        return observation, reward, done, info

    def _apply_action(self, action):
        # Implement how the action is applied in SUMO (e.g., change traffic light states, vehicle behavior)
        pass

    def _get_observation(self):
        # Retrieve the current state from SUMO (e.g., vehicle positions, traffic lights)
        return np.random.rand(10)  # Example: return random observation (replace with actual data)

    def _calculate_reward(self):
        # Implement reward calculation based on current state (e.g., minimize waiting time, collisions, etc.)
        return 1.0  # Example: return a constant reward (replace with actual logic)

    def _check_done(self):
        # Check if the episode is done (e.g., after a certain number of steps or a condition in SUMO)
        return traci.simulation.getMinExpectedNumber() <= 0

    def _get_initial_observation(self):
        # Get the initial state when resetting the environment
        return self._get_observation()

    def render(self, mode='human'):
        # Optional: Implement rendering (usually for debugging)
        pass

    def close(self):
        # Close the SUMO simulation
        traci.close()

# Step 2: Define the Custom Model (if using a custom model, optional)

class CustomTorchModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # Simple feed-forward network with two hidden layers
        self.fc1 = nn.Linear(obs_space.shape[0], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_out = nn.Linear(256, num_outputs)

    def forward(self, input_dict, state, seq_lens):
        x = torch.relu(self.fc1(input_dict["obs"]))
        x = torch.relu(self.fc2(x))
        return self.fc_out(x), state

    def value_function(self):
        return torch.zeros(1)

# Register the custom model (if using)
ModelCatalog.register_custom_model("my_model", CustomTorchModel)

# Step 3: Configure RLlib

config = {
    "env": SumoEnv,  # Register the custom SUMO environment
    "framework": "torch",  # Can also be "tf" for TensorFlow
    "model": {
        "custom_model": "my_model",  # Use custom model (optional)
        "vf_share_layers": True,  # Share layers for value function
    },
    "num_workers": 1,  # Number of workers to use for parallelism
    "env_config": {},  # Pass any necessary environment configuration here
    "action_space": spaces.Discrete(5),  # Match the environment's action space
    "train_batch_size": 4000,  # Adjust based on your resource constraints
    "sgd_minibatch_size": 128,
    "num_sgd_iter": 30,
    "rollout_fragment_length": 200,
    "lr": 0.0003,
}

# Step 4: Train the Model

if __name__ == "__main__":
    # Initialize the PPO trainer
    trainer = PPOTrainer(env=SumoEnv, config=config)

    # Training loop
    for i in range(100):  # Train for 100 iterations (you can adjust this)
        result = trainer.train()
        print(f"Iteration {i}: reward = {result['episode_reward_mean']}")

        # Optionally, save checkpoints
        if i % 10 == 0:
            checkpoint = trainer.save("D:/PBL-2_FInal")
            print(f"Checkpoint saved at {checkpoint}")
