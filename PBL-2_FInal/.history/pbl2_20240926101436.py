import gym
import numpy as np
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.utils import try_import_tf
from ray.rllib.env import MultiAgentEnv
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.algorithms import AlgorithmConfig
import traci  # Ensure you have TraCI installed for SUMO

tf = try_import_tf()

class YourCustomEnv(MultiAgentEnv):
    def __init__(self, config):
        # Initialize your environment here
        self.num_agents = config.get("num_agents", 2)  # Set the number of agents
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)  # Adjust shape as needed
        self.action_space = gym.spaces.Discrete(3)  # Adjust action space according to your task

        # Initialize SUMO
        sumo_net_file = config.get("sumo_net_file", "D:\PBL-2_FInal\map.net.xml")
        sumo_route_file = config.get("sumo_route_file", "D:\PBL-2_FInal\generated_routes.rou.alt.xml")
        self.sumo_cmd = ["sumo-gui", "-c", sumo_route_file, "--net-file", sumo_net_file]
        
        # Start the SUMO simulation
        traci.start(self.sumo_cmd)

    def reset(self):
        # Reset the SUMO simulation and return initial observations
        traci.load(["-c", "D:\PBL-2_FInal\map.sumocfg"])  # Load the SUMO config
        return {f"agent_{i}": self.observation_space.sample() for i in range(self.num_agents)}

    def step(self, action_dict):
        # Implement your step logic here
        traci.simulationStep()  # Advance the SUMO simulation
        obs = {f"agent_{i}": self.observation_space.sample() for i in range(self.num_agents)}  # Get observations
        rewards = {f"agent_{i}": 1.0 for i in range(self.num_agents)}  # Dummy rewards
        done = {f"agent_{i}": False for i in range(self.num_agents)}  # Define when episodes end
        return obs, rewards, done, {}

    def close(self):
        # Close the SUMO simulation
        traci.close()

# Configuration for the PPO trainer
ppo_config = AlgorithmConfig().environment(
    env=YourCustomEnv,
    env_config={
        "num_agents": 3,  # Adjust config as needed
        "sumo_net_file": "D:/PBL-2_FInal/map.net.xml",  # Path to your .net.xml file
        "sumo_route_file": "D:/PBL-2_FInal/generated_routes.rou.alt.xml",  # Path to your routes file
    },
).rollouts(
    num_rollout_workers=2,  # Number of workers for parallel rollout
).training(
    train_batch_size=4000,
    model={
        "fcnet_hiddens": [256, 256],
        "fcnet_activation": "relu",
    },
).framework("tf").resources(
    num_gpus=0,  # Set to 1 if you have a GPU
)

# Build the PPO trainer
ppo_trainer = ppo_config.build()

# Training loop
for i in range(10):  # Change the number of iterations as needed
    result = ppo_trainer.train()
    print(f"Iteration {i}: {result['episode_reward_mean']}")

# Save the trained model
ppo_trainer.save("D:/PBL-2_FInal/")  # Specify your path to save the model
