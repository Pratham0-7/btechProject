import gym
import numpy as np
import sumo_gym
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.policy.policy import PolicySpec

# Make sure to replace 'sumo_gym:sumo-v0' with the correct environment ID
# Check if 'sumo_gym' is properly installed and registered

class MyEnv(MultiAgentEnv):
    def __init__(self, env_config):
        # Use the correct environment ID for your SUMO gym
        self.env = gym.make('sumo_gym:sumo-v0', 
                             sumo_net_file='D:/PBL-2_FInal/map.net.xml',
                             route_file='D:/PBL-2_FInal/generated_routes.rou.alt.xml', 
                             num_vehicles=10)  # Modify num_vehicles as needed

    def reset(self):
        return self.env.reset()

    def step(self, action_dict):
        observations, rewards, dones, infos = self.env.step(action_dict)
        return observations, rewards, dones, infos

# Register the environment
tune.register_env("MyEnv", lambda config: MyEnv(config))

# Configuration for PPO
ppo_config = (
    PPOConfig()
    .environment("MyEnv")
    .framework("torch")
    .rollouts(num_rollout_workers=1)  # Update to use num_env_runners if needed
    .training(lr=1e-4, model={"fcnet_hiddens": [128, 128], "fcnet_activation": "relu"})
    .resources(num_gpus=0)
    .multi_agent(
        policies={
            "default_policy": PolicySpec(
                observation_space=gym.spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32),
                action_space=gym.spaces.Discrete(4)  # Assuming 4 actions: forward, left, right, stop
            )
        },
        policy_mapping_fn=lambda agent_id: "default_policy"
    )
)

# Build the PPO Trainer
ppo_trainer = ppo_config.build()

# Training loop
for i in range(100):  # Modify the range as needed
    result = ppo_trainer.train()
    print(f"Iteration: {i}, Reward: {result['episode_reward_mean']}")
