import ray
print("Ray version:", ray.__version__)

from ray.rllib.agents.ppo import PPOTrainer
print("PPOTrainer imported successfully!")
