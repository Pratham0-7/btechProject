import ray

print("Ray version:", ray.__version__)

# Importing PPO from the new path
from ray.rllib.algorithms.ppo import PPO

print("PPO imported successfully!")
