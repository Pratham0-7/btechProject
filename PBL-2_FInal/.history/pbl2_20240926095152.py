import os
import ray
import subprocess
import gym
import numpy as np
import traci  # for SUMO integration
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.multi_agent_env import MultiAgentEnv  # Import this to define custom MultiAgentEnv
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.utils import try_import_tf
import warnings

# Suppress DeprecationWarnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Define the custom SUMO multi-agent environment
class YourSumoEnv(MultiAgentEnv):
    def __init__(self, env_config=None):
        super().__init__()
        # SUMO files and settings
        self.sumo_net_file = "D:\\PBL-2_FInal\\map.net.xml"
        self.sumo_route_file = "D:\\PBL-2_FInal\\generated_routes.rou.alt.xml"
        self.sumo_binary = "sumo-gui"  # Change to "sumo" if you want to run headless
        self.sumo_cmd = [
            self.sumo_binary,
            "-n", self.sumo_net_file,
            "-r", self.sumo_route_file,
            "--step-length", "0.1",
            "--no-warnings", "true"
        ]

        # Define the action and observation spaces
        self.action_space = gym.spaces.Discrete(2)  # 0: No action, 1: Move to nearest charging station
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)

        self.current_step = 0
        self.sumo_process = None

    # Start SUMO simulation
    def reset(self):
        print("Resetting environment and starting SUMO...")
        if self.sumo_process:
            self.sumo_process.terminate()
            self.sumo_process.wait()

        # Launch SUMO with TraCI
        traci.start(self.sumo_cmd)
        self.current_step = 0
        print("SUMO started successfully.")
        return self.get_observation()

    # Get vehicle observations using TraCI
    def get_observation(self):
        traci.simulationStep()  # Advance SUMO one step
        vehicle_ids = traci.vehicle.getIDList()  # Get all vehicle IDs

        if vehicle_ids:
            # Get data for the first vehicle (in a multi-agent scenario, you'd loop through all vehicles)
            vehicle_id = vehicle_ids[0]
            speed = traci.vehicle.getSpeed(vehicle_id)  # Vehicle speed
            position = traci.vehicle.getPosition(vehicle_id)  # (x, y) coordinates
            battery = traci.vehicle.getParameter(vehicle_id, "device.battery.actualBatteryCapacity")  # Battery level

            # Nearest charging station distance (Assume you have this logic from your SUMO map)
            nearest_charging_station_distance = self.get_distance_to_nearest_station(position)

            # Return the observation: speed, battery level, and distance to the nearest charging station
            return np.array([speed, battery, position[0], position[1], nearest_charging_station_distance], dtype=np.float32)
        else:
            return np.zeros(5)  # No vehicles in simulation

    # Calculate distance to the nearest charging station (stub for actual calculation logic)
    def get_distance_to_nearest_station(self, position):
        # Implement logic to calculate distance to the nearest charging station using map data
        return np.random.random() * 100  # Placeholder distance (replace with actual computation)

    # Define reward logic based on vehicle state
    def calculate_reward(self, action):
        vehicle_ids = traci.vehicle.getIDList()

        if vehicle_ids:
            vehicle_id = vehicle_ids[0]
            battery = traci.vehicle.getParameter(vehicle_id, "device.battery.actualBatteryCapacity")
            speed = traci.vehicle.getSpeed(vehicle_id)

            # Reward logic:
            # 1. Reward for reaching a charging station
            # 2. Penalty for low battery
            # 3. Penalty for being far from a charging station

            nearest_charging_station_distance = self.get_distance_to_nearest_station(traci.vehicle.getPosition(vehicle_id))

            reward = 0.0
            if action == 1 and nearest_charging_station_distance < 10:  # Assume vehicle reaches charging station
                reward += 10  # Reward for reaching a station
            if battery < 20:  # Penalty for low battery
                reward -= 5
            reward -= nearest_charging_station_distance / 100  # Penalty for distance from station

            return reward
        else:
            return 0.0

    # Advance the environment by one step and process actions
    def step(self, action):
        print(f"Taking action: {action}")
        self.current_step += 1

        # Perform the action: e.g., navigate to charging station if action is 1
        if action == 1:
            self.navigate_to_charging_station()

        observation = self.get_observation()  # Get updated observation
        reward = self.calculate_reward(action)  # Calculate reward
        done = self.current_step >= 1000  # Episode done if 1000 steps are reached

        print(f"Step: {self.current_step}, Observation: {observation}, Reward: {reward}, Done: {done}")
        return observation, reward, done, {}

    # Stub for navigating to the nearest charging station
    def navigate_to_charging_station(self):
        vehicle_ids = traci.vehicle.getIDList()
        if vehicle_ids:
            vehicle_id = vehicle_ids[0]
            # Implement logic to route vehicle to nearest charging station (e.g., using TraCI rerouting functions)
            print(f"Navigating vehicle {vehicle_id} to nearest charging station")

    # Close the environment and the SUMO process
    def close(self):
        if self.sumo_process:
            self.sumo_process.terminate()
            self.sumo_process.wait()
        traci.close()

# Initialize TensorFlow
print("Initializing TensorFlow...")
tf = try_import_tf()
print("TensorFlow initialized.")

# Function to create the environment
def create_env(env_config):
    return YourSumoEnv(env_config)

# Register the environment with Ray
ray.tune.register_env("YourSumoEnv", create_env)

# Define PPO Configuration
config = (
    PPOConfig()
    .environment("YourSumoEnv")
    .resources(num_gpus=0)
    .training(
        model={
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
        print(f"Iteration: {i}, reward: {result['episode_reward_mean']}")  # Ensure this prints results

    # Save the model checkpoint
    checkpoint = trainer.save()
    print(f"Checkpoint saved at {checkpoint}")

    # Close the environment for all workers
    trainer.workers.foreach_worker(lambda worker: worker.env.close())

# Execute the main function
if __name__ == "__main__":
    print("Starting the main function...")
    main()
