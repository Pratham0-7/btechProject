import gym
from gym import spaces
import numpy as np
import traci
from ray.rllib.env.env_context import EnvContext
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import torch
import torch.nn as nn
from ray import tune
from ray.rllib.algorithms.ppo import PPO

# Step 1: Define the Custom SUMO Environment
class SumoEnv(gym.Env):
    def __init__(self, config: EnvContext):
        super(SumoEnv, self).__init__()

        # Define the action space (e.g., 5 discrete actions)
        self.action_space = spaces.Discrete(5)
        
        # Define the observation space (10 features for example)
        self.observation_space = spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)

        # Initialize SUMO environment
        self._initialize_sumo()

    def _initialize_sumo(self):
        # Start SUMO with TraCI
        sumoBinary = "sumo"  # Use "sumo-gui" if you want visualization
        sumoCmd = ["sumo", "-c", "D:\\PBL-2_FInal\\map.sumocfg"]  # Update with your SUMO config path
        traci.start(sumoCmd)

    def reset(self):
        # Reset SUMO environment and get the initial observation
        traci.close()  # Close any existing connection
        self._initialize_sumo()  # Restart the SUMO process
        return self._get_initial_observation()

    def step(self, action):
        # Apply the action to the environment
        self._apply_action(action)
        
        # Simulate one step in SUMO
        traci.simulationStep()

        # Get the new observation
        observation = self._get_observation()

        # Calculate reward
        reward = self._calculate_reward()

        # Determine if the episode is done
        done = self._check_done()

        # Additional info
        info = {}

        return observation, reward, done, info

    def _apply_action(self, action):
        # Implement how the action is applied in SUMO
        # Example: Changing the speed of a vehicle based on action
        # You need to implement this based on your specific action mapping
        pass  # Implement the logic for the action here

    def _get_observation(self):
        # Retrieve the current state from SUMO
        # Replace with actual state data, for example:
        state_data = np.random.rand(10)  # Placeholder for the actual state
        return state_data

    def _calculate_reward(self):
        # Implement reward calculation logic
        # Placeholder logic; replace with actual computation based on the simulation
        return 1.0  # Example reward value

    def _check_done(self):
        # Check if the episode is done
        return traci.simulation.getMinExpectedNumber() <= 0

    def _get_initial_observation(self):
        # Get the initial state when resetting the environment
        return self._get_observation()

    def render(self, mode='human'):
        # Optional: Implement rendering
        pass

    def close(self):
        # Close the SUMO simulation
        traci.close()


# Step 2: Define the Custom Model
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
        return torch.zeros(1)


# Register the custom model
ModelCatalog.register_custom_model("my_model", CustomTorchModel)

# Step 3: Configure RLlib
# Create an instance of SumoEnv once
env_instance = SumoEnv({})

# Use the observation_space and action_space defined in SumoEnv
obs_space = env_instance.observation_space
action_space = env_instance.action_space

config = {
    "env": SumoEnv,
    "framework": "torch",
    "model": {
        "custom_model": "my_model",
        "vf_share_layers": True,
    },
    "num_workers": 1,
    "env_config": {},
    "train_batch_size": 4000,
    "sgd_minibatch_size": 128,
    "num_sgd_iter": 30,
    "rollout_fragment_length": 200,
    "lr": 0.0003,
    "multiagent": {
        "policies": {
            "default_policy": (None, obs_space, action_space, {}),
        },
    },
}

# Step 4: Train the Model
if __name__ == "__main__":
    trainer = PPO(env=SumoEnv, config=config)

    # Training loop
    for i in range(100):
        result = trainer.train()
        print(f"Iteration {i}: reward = {result['episode_reward_mean']}")

        if i % 10 == 0:
            checkpoint = trainer.save("D:/PBL-2_FInal")
            print(f"Checkpoint saved at {checkpoint}")
