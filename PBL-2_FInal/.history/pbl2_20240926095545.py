import os
import ray
import subprocess
import gym
import numpy as np
import traci  # for SUMO integration
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.utils import try_import_tf
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import torch
import torch.nn as nn
import warnings

# Suppress DeprecationWarnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Define the custom SUMO multi-agent environment
class YourSumoEnv(MultiAgentEnv):
    def __init__(self, env_config=None):
        super().__init__()
        self.sumo_net_file = "D:\\PBL-2_FInal\\map.net.xml"
        self.sumo_route_file = "D:\\PBL-2_FInal\\generated_routes.rou.alt.xml"
        self.sumo_binary = "sumo-gui"  # Change to "sumo" for headless mode
        self.sumo_cmd = [
            self.sumo_binary,
            "-n", self.sumo_net_file,
            "-r", self.sumo_route_file,
            "--step-length", "0.1",
            "--no-warnings", "true"
        ]

        self.action_space = gym.spaces.Discrete(2)  # 0: No action, 1: Move to nearest charging station
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)

        self.current_step = 0
        self.sumo_process = None

    def reset(self):
        print("Resetting environment and starting SUMO...")
        if self.sumo_process:
            self.sumo_process.terminate()
            self.sumo_process.wait()

        traci.start(self.sumo_cmd)
        self.current_step = 0
        print("SUMO started successfully.")
        return self.get_observation()

    def get_observation(self):
        traci.simulationStep()
        vehicle_ids = traci.vehicle.getIDList()

        if vehicle_ids:
            vehicle_id = vehicle_ids[0]
            speed = traci.vehicle.getSpeed(vehicle_id)
            position = traci.vehicle.getPosition(vehicle_id)
            battery = traci.vehicle.getParameter(vehicle_id, "device.battery.actualBatteryCapacity")
            nearest_charging_station_distance = self.get_distance_to_nearest_station(position)

            return np.array([speed, battery, position[0], position[1], nearest_charging_station_distance], dtype=np.float32)
        else:
            return np.zeros(5)

    def get_distance_to_nearest_station(self, position):
        return np.random.random() * 100  # Placeholder logic

    def calculate_reward(self, action):
        vehicle_ids = traci.vehicle.getIDList()

        if vehicle_ids:
            vehicle_id = vehicle_ids[0]
            battery = traci.vehicle.getParameter(vehicle_id, "device.battery.actualBatteryCapacity")
            nearest_charging_station_distance = self.get_distance_to_nearest_station(traci.vehicle.getPosition(vehicle_id))

            reward = 0.0
            if action == 1 and nearest_charging_station_distance < 10:
                reward += 10
            if battery < 20:
                reward -= 5
            reward -= nearest_charging_station_distance / 100

            return reward
        else:
            return 0.0

    def step(self, action):
        print(f"Taking action: {action}")
        self.current_step += 1

        if action == 1:
            self.navigate_to_charging_station()

        observation = self.get_observation()
        reward = self.calculate_reward(action)
        done = self.current_step >= 1000

        print(f"Step: {self.current_step}, Observation: {observation}, Reward: {reward}, Done: {done}")
        return observation, reward, done, {}

    def navigate_to_charging_station(self):
        vehicle_ids = traci.vehicle.getIDList()
        if vehicle_ids:
            vehicle_id = vehicle_ids[0]
            print(f"Navigating vehicle {vehicle_id} to nearest charging station")

    def close(self):
        if self.sumo_process:
            self.sumo_process.terminate()
            self.sumo_process.wait()
        traci.close()

# Define Custom Torch Model
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
        return self.fc[-1].weight  # Placeholder for value function

# Initialize TensorFlow
print("Initializing TensorFlow...")
tf = try_import_tf()
print("TensorFlow initialized.")

# Function to create the environment
def create_env(env_config):
    return YourSumoEnv(env_config)

# Register the environment and custom model with Ray
ray.tune.register_env("YourSumoEnv", create_env)
ModelCatalog.register_custom_model("custom_torch_model", CustomTorchModel)

# Define PPO Configuration
config = (
    PPOConfig()
    .environment("YourSumoEnv")
    .resources(num_gpus=0)
    .training(
        model={
            "custom_model": "custom_torch_model",
            "fcnet_hiddens": [64, 64],
            "fcnet_activation": "tanh",
        }
    )
)

# Add multi-agent policy to the configuration
def add_multi_agent_policy(config):
    config.multi_agent(
        policies={
            "default_policy": PolicySpec(
                observation_space=YourSumoEnv().observation_space,
                action_space=YourSumoEnv().action_space,
            )
        }
    )

add_multi_agent_policy(config)

# Initialize Ray
print("Initializing Ray...")
ray.init()

# Main training loop
def main():
    print("Initializing trainer...")
    trainer = config.build()  # Build the trainer

    print("Starting training loop...")
    for i in range(100):  # Train for 100 iterations
        result = trainer.train()  # Perform training
        print(f"Iteration: {i}, reward: {result['episode_reward_mean']}")

    # Save the model checkpoint
    checkpoint = trainer.save()
    print(f"Checkpoint saved at {checkpoint}")

    # Close the environment for all workers
    trainer.workers.foreach_worker(lambda worker: worker.env.close())

# Execute the main function
if __name__ == "__main__":
    print("Starting the main function...")
    main()
