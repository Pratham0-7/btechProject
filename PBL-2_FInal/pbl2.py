from ray import tune  # Import tune from Ray
from gym.envs.registration import register
import gym
from gym import spaces
import numpy as np

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
        
