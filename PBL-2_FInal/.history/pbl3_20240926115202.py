import gym
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig

# Define your custom environment class
class MyEnv:
    def __init__(self, config):
        self.config = config
        try:
            # Ensure you're using the correct environment name here
            self.env = gym.make("sumo_gym:sumo-v0", config=config)
        except Exception as e:
            print(f"Failed to create environment: {e}")

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def close(self):
        self.env.close()

# Register the environment
tune.register_env("MyEnv", lambda config: MyEnv(config))

def train_my_env():
    config = PPOConfig().environment(
        "MyEnv"  # Use the registered environment name here
    ).framework("torch").rollouts(num_rollout_workers=2)  # Adjust as needed

    agent = config.build()
    
    # Example training loop
    for i in range(10):  # Replace with your desired number of training iterations
        results = agent.train()
        print(f"Iteration {i}: {results}")

if __name__ == "__main__":
    # Print registered environments
    try:
        registered_envs = gym.envs.registry.all()
        print("Registered environments:")
        for env in registered_envs:
            print(env.id)  # Print the environment IDs only
    except Exception as e:
        print(f"Error retrieving registered environments: {e}")
        
    train_my_env()
