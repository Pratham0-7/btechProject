import gym
from gym import spaces
from stable_baselines3 import PPO
from sumo_gym import SumoGymEnv  # Import the correct class

class SUMOEnv(SumoGymEnv):
    def __init__(self):
        # Initialize the SUMO environment with your configuration
        super(SUMOEnv, self).__init__(config='map.sumocfg')  # Use your actual config file
        self.action_space = spaces.Discrete(2)  # Example: two actions
        self.observation_space = spaces.Box(low=0, high=1, shape=(4,), dtype=float)

    def reset(self):
        return super().reset()  # Reset using the parent class's reset

    def step(self, action):
        obs, reward, done, info = super().step(action)  # Call the parent class's step
        return obs, reward, done, info

# Main training loop
if __name__ == "__main__":
    env = SUMOEnv()
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
