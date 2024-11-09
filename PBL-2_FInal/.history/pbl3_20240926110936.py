import os
import ray
import subprocess
import gym
import numpy as np
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.utils import try_import_tf
from gym.envs.registration import register
from gym import spaces

# Register the custom environment
def register_env():
    register(
        id='sumo_gym:sumo-v0',
        entry_point='sumo_gym.envs:SumoEnv',  # Ensure this points to the correct module and class
        max_episode_steps=1000
    )
    tune.register_env("MyEnv", lambda config: MyEnv(config))

register_env()

class MyEnv(gym.Env):
    def __init__(self, config):
        super(MyEnv, self).__init__()
        # Define the observation space (example: Box space for continuous observations)
        self.observation_space = spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)
        
        # Define the action space (example: Discrete space for a finite set of actions)
        self.action_space = spaces.Discrete(2)

        # Initialize your environment here
        self.env = gym.make('sumo_gym:sumo-v0', config=config)
        
    def reset(self):
        # Reset the environment state
        return self.env.reset()

    def step(self, action):
        # Execute the action in the environment
        return self.env.step(action)

    def render(self, mode='human'):
        # Render the environment if needed
        return self.env.render(mode=mode)

    def close(self):
        self.env.close()


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
        self.reset()

    def reset(self):
        self.sumo_process = subprocess.Popen(self.sumo_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.current_step = 0
        return self.get_observation()

    def step(self, action):
        self.current_step += 1
        observation = self.get_observation()
        reward = self.calculate_reward(action)
        done = self.current_step >= 1000
        print(f"Step: {self.current_step}, Action: {action}, Observation: {observation}, Reward: {reward}, Done: {done}")
        return observation, reward, done, {}

    def get_observation(self):
        return np.random.random(10)

    def calculate_reward(self, action):
        return 1.0

    def close(self):
        if self.sumo_process:
            self.sumo_process.terminate()
            self.sumo_process.wait()

tf = try_import_tf()

def create_env(env_config):
    return YourSumoEnv(env_config)

# Register YourSumoEnv with Ray Tune
tune.register_env("YourSumoEnv", create_env)

# Define the PPO configuration
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

# Multi-agent setup for the policies
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
    # Build the trainer using the defined configuration
    trainer = config.build()
    
    # Run training for 100 iterations
    for i in range(100):
        result = trainer.train()
        print(f"Iteration: {i}, reward: {result['episode_reward_mean']}")
    
    # Save the trained model checkpoint
    checkpoint = trainer.save()
    print(f"Checkpoint saved at {checkpoint}")
    
    # Close the environment after training
    trainer.workers.foreach_worker(lambda worker: worker.env.close())

if __name__ == "__main__":
    main()
