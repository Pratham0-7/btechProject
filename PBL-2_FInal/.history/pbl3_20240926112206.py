from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
import gym
import numpy as np

# Define the MyEnv class and register the environment here (not shown for brevity)

def train(env_name):
    # Configure the PPO algorithm
    config = (
        PPOConfig()
        .environment(env_name)  # Use your registered environment
        .resources(num_gpus=0)  # Adjust based on your hardware availability
        .training(
            train_batch_size=4000,  # Total batch size for training
            num_sgd_iter=10,        # Number of epochs to perform
            clip_param=0.3,         # Clipping parameter for PPO
            gamma=0.99,             # Discount factor
            lambda_=0.95            # Lambda for GAE
        )
    )

    # Create the trainer
    trainer = config.build()

    # Training loop
    for i in range(10):  # Adjust the number of iterations as necessary
        result = trainer.train()  # Perform training
        print(f"Iteration {i}: reward {result['episode_reward_mean']}")  # Print mean reward

    # Save the trained model
    checkpoint = trainer.save("D:/PBL-2_FInal/model/checkpoint")  # Specify your save path
    print(f"Model saved at {checkpoint}")  # Confirm save location

# Testing Function
def load_model(model_path):
    # Load the agent from the checkpoint
    agent = PPOConfig().from_checkpoint(model_path)  # Corrected this line
    return agent

def test(agent, env_name, num_episodes=10):
    env = gym.make(env_name)
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.compute_action(state)  # Get action from the agent
            state, reward, done, info = env.step(action)  # Take a step in the environment
            total_reward += reward
            env.render()  # Optional: render the environment

        print(f"Episode {episode + 1}: Total Reward: {total_reward}")

    env.close()  # Close the environment after testing

# Main Execution
if __name__ == "__main__":
    env_name = "MyEnv"

    # Train the model
    train(env_name)

    # Test the trained model
    model_path = "D:/PBL-2_FInal/model/checkpoint"  # Specify your saved model path
    agent = load_model(model_path)
    test(agent, env_name)
