import ray
print("Ray version:", ray.__version__) 
from ray.rllib.algorithms.ppo import PPO
print("PPOTrainer imported successfully!")
