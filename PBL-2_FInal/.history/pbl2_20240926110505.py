from gym.envs.registration import register

# Register the custom environment
def register_env():
    register(
        id='sumo_gym:sumo-v0',
        entry_point='sumo_gym.envs:SumoEnv',  # Ensure this points to the correct module and class
        max_episode_steps=1000
    )
    tune.register_env("MyEnv", lambda config: MyEnv(config))

register_env()
