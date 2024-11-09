# import gym
# from gym import envs

# # Registering the environment
# gym.envs.registration.register(
#     id='sumo_gym:sumo-v0',
#     entry_point='sumo_gym:SumoEnv',  # Ensure this points to the correct environment class
#     max_episode_steps=1000,  # Set max steps per episode if needed
# )

# # Write output to a file
# with open('output.txt', 'w', encoding='utf-8') as f:
#     for env in gym.envs.registry.all():
#         f.write(f"{env}\n")
