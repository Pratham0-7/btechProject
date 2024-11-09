from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
import gym
import numpy as np

# Define the MyEnv class and register the environment...

def train(env_name):
    # Configure the PPO algorithm
    config = (
        PPOConfig()
        .environment(env_name)  # Use your registered environment
        .resources(num_gpus=0)  # Adjust based on your hardware availability
        .training(
            train_batch_size=4000,  # Total batch size for training
            num_sgd_iter=10,        # Number of epochs to perform
            rollouts_per_batch=1,   # Number of rollouts to sample per training iteration
            clip_param=0.3,         # Clipping parameter for PPO
            gamma=0.99,             # Discount factor
            lambda_=0.95            # Lambda for GAE
        )
        .logging(
            log_level="INFO"  # Log level for output
        )
    )

    # Create the trainer
    trainer = config.build()

    # Training loop
    for i in range(10):  # Adjust the number of iterations as necessary
        result = trainer.train()  # Perform training
        print(f"Iteration {i}: reward {result['episode_reward_mean']}")  # Print mean reward

    # Save the trained model
    checkpoint = trainer.save("path/to/save/model")  # Specify your save path
    print(f"Model saved at {checkpoint}")  # Confirm save location


# Testing Function
def load_model(model_path):
    agent = PPOConfig.from_checkpoint(model_path)
    return agent

def test(agent, env_name, num_episodes=10):
    env = gym.make(env_name)
    # Testing loop...

# Main Execution
if __name__ == "__main__":
    env_name = "MyEnv"

    # Train the model
    train(env_name)

    # Test the trained model
    model_path = "D:/PBL-2_FInal/model/"  # Specify your saved model path
    agent = load_model(model_path)
    test(agent, env_name)
