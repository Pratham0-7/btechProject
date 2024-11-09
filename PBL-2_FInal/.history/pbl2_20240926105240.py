import gym
from gym import envs

# Registering the environment
gym.envs.registration.register(
    id='sumo_gym:sumo-v0',
    entry_point='sumo_gym:SumoEnv',  # Ensure this points to the correct environment class
    max_episode_steps=1000,  # Set max steps per episode if needed
)
