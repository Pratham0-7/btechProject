import os
import traci  # SUMO Python API
import ray
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.registry import register_env
from gym.spaces import Discrete, Box
import numpy as np

# Path to SUMO binary
sumo_binary = "sumo"  # or "sumo-gui" for visualization
sumo_config = r"D:\path\to\your\map.sumocfg"  # Update with your sumo config file

class SumoEnv(MultiAgentEnv):
    def __init__(self, config):
        self.step_length = 0.1  # SUMO simulation step time
        self.sumo_cmd = [sumo_binary, "-c", sumo_config, "--step-length", str(self.step_length)]
        self.vehicles = []
        self.current_step = 0

        # Define observation space (e.g., vehicle speed, position, etc.)
        self.observation_space = Box(low=0, high=1, shape=(5,), dtype=np.float32)  # Modify as per observation needs
        self.action_space = Discrete(5)  # Discrete actions like accelerate, brake, etc.
        
    def reset(self):
        """Resets the SUMO simulation and the environment"""
        traci.start(self.sumo_cmd)
        self.vehicles = traci.vehicle.getIDList()
        self.current_step = 0
        
        # Initialize observations
        observations = {}
        for vehicle_id in self.vehicles:
            observations[vehicle_id] = self.get_observation(vehicle_id)
        return observations

    def step(self, actions):
        """Executes the given actions and advances the SUMO simulation"""
        rewards = {}
        dones = {}
        infos = {}
        for vehicle_id, action in actions.items():
            self.apply_action(vehicle_id, action)

        traci.simulationStep()  # Advance simulation step
        self.vehicles = traci.vehicle.getIDList()

        # Gather new observations and rewards
        observations = {}
        for vehicle_id in self.vehicles:
            observations[vehicle_id] = self.get_observation(vehicle_id)
            rewards[vehicle_id] = self.compute_reward(vehicle_id)
            dones[vehicle_id] = False  # Set to True if episode ends for any vehicle
            infos[vehicle_id] = {}

        self.current_step += 1
        dones["__all__"] = self.current_step >= 1000  # End simulation after 1000 steps
        return observations, rewards, dones, infos

    def apply_action(self, vehicle_id, action):
        """Apply the agent's action to control the vehicle in SUMO"""
        if action == 0:  # Accelerate
            traci.vehicle.setSpeed(vehicle_id, traci.vehicle.getAllowedSpeed(vehicle_id) + 2.0)
        elif action == 1:  # Brake
            traci.vehicle.setSpeed(vehicle_id, 0)
        # Add more actions as needed

    def get_observation(self, vehicle_id):
        """Get the current observation for a vehicle"""
        speed = traci.vehicle.getSpeed(vehicle_id)
        pos = traci.vehicle.getPosition(vehicle_id)
        lane_id = traci.vehicle.getLaneID(vehicle_id)
        # Example observation: speed, x, y, and lane_id as categorical value
        return np.array([speed, pos[0], pos[1], lane_id, 1.0])  # Adjust as necessary

    def compute_reward(self, vehicle_id):
        """Compute reward for a vehicle"""
        speed = traci.vehicle.getSpeed(vehicle_id)
        # Example reward: maximize speed, avoid traffic lights
        return speed / traci.vehicle.getAllowedSpeed(vehicle_id)  # Reward normalized by max speed

    def close(self):
        """Close the SUMO simulation"""
        traci.close()

# Register the environment in Ray
def env_creator(env_config):
    return SumoEnv(env_config)

register_env("sumo_env", env_creator)

# Configuration for RLlib
config = {
    "env": "sumo_env",
    "multiagent": {
        "policies": {
            "default_policy": (None, Box(low=0, high=1, shape=(5,)), Discrete(5), {}),
        },
        "policy_mapping_fn": lambda agent_id: "default_policy",
    },
    "framework": "torch",  # or "tf"
    "num_workers": 1,
}

# Initialize Ray and start training
ray.init()
from ray.rllib.algorithms.ppo import PPO

trainer = PPO(config=config)
for i in range(100):  # Train for 100 iterations
    result = trainer.train()
    print(f"Iteration {i}: reward = {result['episode_reward_mean']}")

trainer.save("/path/to/checkpoint")

ray.shutdown()
