from ray import tune  # Import tune from Ray
from gym.envs.registration import register
import gym
from gym import spaces
import numpy as np
import ray
from ray.rllib.algorithms.ppo import PPOConfig


# Define the custom environment class
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

# Register the custom environment
def register_env():
    register(
        id='MyEnv-v0',  # Use a unique ID for your environment
        entry_point=__name__ + ':MyEnv',  # Points to the MyEnv class in the current module
        max_episode_steps=1000
    )
    tune.register_env("MyEnv", lambda config: MyEnv(config))

register_env()

def train_my_env(num_episodes=1000, num_workers=1):
    # Initialize Ray
    ray.init()

    # Create a PPO configuration
    config = (
        PPOConfig()
        .environment("MyEnv")  # Specify your registered environment
        .rollouts(num_rollout_workers=num_workers)  # Configure rollouts with the number of workers
        .framework("torch")  # Use PyTorch (or "tf" for TensorFlow)
    )

    # Build the trainer with the specified configuration
    agent = config.build()

    for episode in range(num_episodes):
        result = agent.train()  # Train for one iteration
        print(f"Episode {episode + 1}: reward={result['episode_reward_mean']:.2f}")

    # Save the trained model
    agent.save("ppo_model")

    # Shutdown Ray
    ray.shutdown()

# Uncomment the line below to train your environment
# train_my_env()
