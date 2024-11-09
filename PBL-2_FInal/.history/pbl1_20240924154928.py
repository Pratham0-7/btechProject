import os
import ray
import subprocess
import gym
import numpy as np
import traci  # SUMO's TraCI interface
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.utils import try_import_tf

class YourSumoEnv(MultiAgentEnv):
    def __init__(self, env_config=None):
        super().__init__()
        self.sumo_net_file = "D:\\PBL-2_FInal\\map.net.xml"
        self.sumo_route_file = "D:\\PBL-2_FInal\\generated_routes.rou.alt.xml"
        self.sumo_binary = "sumo-gui"
        self.sumo_cmd = [self.sumo_binary, "-n", self.sumo_net_file, "-r", self.sumo_route_file,
                         "--step-length", "0.1", "--no-warnings", "true"]

        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)

        self.sumo_process = None
        self.vehicles = {}  # Dictionary to store vehicle positions over time
        self.reset()

    def reset(self):
        self.sumo_process = subprocess.Popen(self.sumo_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.current_step = 0
        traci.start([self.sumo_binary, "-n", self.sumo_net_file, "-r", self.sumo_route_file])
        return self.get_observation()

    def step(self, action):
        traci.simulationStep()  # Advance the simulation step
        self.current_step += 1

        # Update vehicle positions
        self.update_vehicle_positions()

        # Print path predictions for all vehicles
        self.print_predicted_paths()

        observation = self.get_observation()
        reward = self.calculate_reward(action)
        done = self.current_step >= 1000
        print(f"Step: {self.current_step}, Action: {action}, Observation: {observation}, Reward: {reward}, Done: {done}")
        return observation, reward, done, {}

    def update_vehicle_positions(self):
        # Get all vehicle IDs in the simulation
        vehicle_ids = traci.vehicle.getIDList()
        for vehicle_id in vehicle_ids:
            # Get the vehicle's current position (x, y)
            x, y = traci.vehicle.getPosition(vehicle_id)
            if vehicle_id not in self.vehicles:
                self.vehicles[vehicle_id] = []
            # Append the new position to the vehicle's trajectory history
            self.vehicles[vehicle_id].append((x, y))

    def get_observation(self):
        # Example: Return a random observation
        return np.random.random(10)

    def calculate_reward(self, action):
        return 1.0

    def predict_vehicle_path(self, vehicle_id, steps_ahead=10):
        if vehicle_id not in self.vehicles or len(self.vehicles[vehicle_id]) < 2:
            return None  # Not enough data to predict

        # Get the last two positions of the vehicle
        (x1, y1), (x2, y2) = self.vehicles[vehicle_id][-2:]

        # Calculate velocity vector (dx, dy)
        dx = x2 - x1
        dy = y2 - y1

        predicted_path = []
        for i in range(1, steps_ahead + 1):
            # Predict future positions by continuing in the direction of the velocity vector
            predicted_x = x2 + i * dx
            predicted_y = y2 + i * dy
            predicted_path.append((predicted_x, predicted_y))

        return predicted_path

    def print_predicted_paths(self):
        for vehicle_id in self.vehicles:
            predicted_path = self.predict_vehicle_path(vehicle_id)
            if predicted_path:
                print(f"Predicted path for vehicle {vehicle_id}: {predicted_path}")

    def close(self):
        if self.sumo_process:
            self.sumo_process.terminate()
            self.sumo_process.wait()
        traci.close()

tf = try_import_tf()

def create_env(env_config):
    return YourSumoEnv(env_config)

ray.tune.register_env("YourSumoEnv", create_env)

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

ray.init()

def main():
    trainer = config.build()
    for i in range(100):
        result = trainer.train()
        print(f"Iteration: {i}, reward: {result['episode_reward_mean']}")
    checkpoint = trainer.save()
    print(f"Checkpoint saved at {checkpoint}")
    trainer.workers.foreach_worker(lambda worker: worker.env.close())

if __name__ == "__main__":
    main()
