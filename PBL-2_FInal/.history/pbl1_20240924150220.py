import os
import ray
import subprocess
import gym
import numpy as np
import traci  # Import TraCI to control SUMO
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
        self.vehicle_id = "your_vehicle_id"  # Update this to your actual vehicle ID
        self.reset()

    def reset(self):
        # Start the SUMO simulation using the subprocess module
        self.sumo_process = subprocess.Popen(self.sumo_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.current_step = 0

        # Initialize TraCI connection to interact with the SUMO simulation
        traci.start([self.sumo_binary, "-n", self.sumo_net_file, "-r", self.sumo_route_file])
        
        return self.get_observation()

    def step(self, action):
        self.current_step += 1

        # Apply the action to the environment (this would depend on your action space setup)
        # Normally, TraCI functions would be called here to control vehicles (e.g., set vehicle speed)
        traci.simulationStep()

        observation = self.get_observation()
        reward = self.calculate_reward(action)
        done = self.current_step >= 1000

        print(f"Step: {self.current_step}, Action: {action}, Observation: {observation}, Reward: {reward}, Done: {done}")
        return observation, reward, done, {}

    def get_observation(self):
        # For now, we'll return random observations; in practice, you'd extract features from SUMO/TraCI
        return np.random.random(10)

    def calculate_reward(self, action):
        # Fetch the vehicle's current speed
        speed = traci.vehicle.getSpeed(self.vehicle_id)

        # Fetch the current distance traveled
        distance_traveled = traci.vehicle.getDistance(self.vehicle_id)

        # Define speed limits and optimal speed range (example)
        max_speed = 30.0  # Max speed in meters/second
        optimal_speed = 15.0  # Optimal cruising speed

        # 1. Reward for speed maintenance (positive if speed is within an optimal range)
        speed_reward = 1.0 - abs(optimal_speed - speed) / optimal_speed

        # 2. Reward for distance traveled (proportional to distance covered)
        distance_reward = distance_traveled / 100  # Scaling down by a factor (example)

        # 3. Penalize for being too slow or fast
        if speed < 5:  # Too slow
            speed_penalty = -1.0
        elif speed > max_speed:  # Too fast
            speed_penalty = -2.0
        else:
            speed_penalty = 0.0

        # 4. Collision Penalty
        collision_info = traci.simulation.getCollidingVehiclesIDList()
        if self.vehicle_id in collision_info:
            collision_penalty = -10.0  # Large penalty for collision
        else:
            collision_penalty = 0.0

        # 5. Fuel Penalty (assuming TraCI can provide fuel consumption)
        fuel_consumption = traci.vehicle.getFuelConsumption(self.vehicle_id)
        fuel_penalty = -fuel_consumption / 100  # Penalize high fuel consumption

        # Sum up the reward components
        total_reward = speed_reward + distance_reward + speed_penalty + collision_penalty + fuel_penalty

        return total_reward

    def close(self):
        # Close SUMO and TraCI when done
        if self.sumo_process:
            self.sumo_process.terminate()
            self.sumo_process.wait()

        # Close the TraCI connection
        traci.close()


# RLlib and PPO setup
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
