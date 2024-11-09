import gym
from gym import spaces
from stable_baselines3 import PPO
import numpy as np
from sumo_gym.envs import FMPEnv

class CustomSUMOEnv(FMPEnv):
    def __init__(self, mode='train'):
        super(CustomSUMOEnv, self).__init__(mode=mode, config_file='map.sumocfg')  # Specify the mode
        self.action_space = spaces.Discrete(2)  # Example: two actions
        self.observation_space = spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)

    def reset(self):
        obs = super().reset()  # Call the reset method from FMPEnv
        return self._get_observation()  # Return the current observation

    def step(self, action):
        obs, reward, done, info = super().step(action)  # Step in the environment
        return self._get_observation(), reward, done, info  # Return the updated observation

    def _get_observation(self):
        # Implement this method to return your observation logic
        return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)  # Replace with actual observation logic

    def _compute_reward(self):
        # Implement your reward logic based on the state of the environment
        return 0  # Replace with actual reward logic

# Main training loop
if __name__ == "__main__":
    env = CustomSUMOEnv(mode='train')  # Specify 'train' mode
    model = PPO("MlpPolicy", env, verbose=1)

    # Train the model
    model.learn(total_timesteps=10000)

    # Evaluate the model
    obs = env.reset()
    for _ in range(1000):
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        if done:
            obs = env.reset()
