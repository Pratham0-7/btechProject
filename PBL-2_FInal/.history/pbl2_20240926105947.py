# import gym
# from gym import envs

# # Registering the environment
# gym.envs.registration.register(
#     id='sumo_gym:sumo-v0',
#     entry_point='sumo_gym:SumoEnv',  # Ensure this points to the correct environment class
#     max_episode_steps=1000,  # Set max steps per episode if needed
# )

# # Write output to a file
# with open('output.txt', 'w', encoding='utf-8') as f:
#     for env in gym.envs.registry.all():
#         f.write(f"{env}\n")

import gym
import sumo_gym
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig

# Initialize Ray
ray.init(ignore_reinit_error=True)

# Step 1: Define the custom environment class
class MyEnv(gym.Env):
    def __init__(self, config):
        super(MyEnv, self).__init__()
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

# Step 2: Register the custom environment
def register_env():
    tune.register_env("MyEnv", lambda config: MyEnv(config))

# Register the environment
register_env()

# Step 3: Create PPO Configuration
ppo_config = (
    PPOConfig()
    .environment("MyEnv")  # Use the registered environment name
    .framework("torch")  # Or "tf" if you prefer TensorFlow
    .rollouts(num_rollout_workers=2)
    .training(
        model={
            "fcnet_hiddens": [64, 64],
            "fcnet_activation": "relu",
        },
        train_batch_size=400,
        # Other training parameters as needed
    )
)

# Step 4: Build the PPO trainer
ppo_trainer = ppo_config.build()

# Example training loop
for i in range(10):
    print(f"Training iteration {i + 1}")
    results = ppo_trainer.train()
    print(f"Training result: {results}")

# Clean up
ppo_trainer.stop()
