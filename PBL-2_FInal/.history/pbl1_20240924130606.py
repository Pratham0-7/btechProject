import gym
from gym import spaces
import numpy as np
import traci
from ray import tune
from ray.rllib.algorithms.ppo import PPO, PPOConfig
import ray

# Step 1: Define the Custom SUMO Environment
class SumoEnv(gym.Env):
    def __init__(self, config):
        super(SumoEnv, self).__init__()

        # Define the action space (5 discrete actions)
        self.action_space = spaces.Discrete(5)
        
        # Define the observation space (10 features for example)
        self.observation_space = spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)

        # Initialize SUMO environment
        self._initialize_sumo()

    def _initialize_sumo(self):
        sumoCmd = ["sumo", "-c", "D:\\PBL-2_FInal\\map.sumocfg"]  # Update with your SUMO config path
        traci.start(sumoCmd)

    def reset(self):
        traci.close()  # Close any existing connection
        self._initialize_sumo()  # Restart the SUMO process
        return self._get_initial_observation()

    def step(self, action):
        self._apply_action(action)
        traci.simulationStep()
        observation = self._get_observation()
        reward = self._calculate_reward()
        done = self._check_done()
        info = {}
        return observation, reward, done, info

    def _apply_action(self, action):
        # Example action mapping (implement your logic)
        pass

    def _get_observation(self):
        # Replace with actual state retrieval logic
        state_data = np.random.rand(10)  # Placeholder for actual state
        return state_data

    def _calculate_reward(self):
        return 1.0  # Placeholder for actual reward calculation

    def _check_done(self):
        return traci.simulation.getMinExpectedNumber() <= 0  # Modify as needed

# Step 2: Configure Ray and the PPO Algorithm
def main():
    ray.init(ignore_reinit_error=True)

    config = PPOConfig().environment(SumoEnv).framework("torch").rollouts(num_rollout_workers=2)

    # Set the parameters for the training
    config = config.training(
        model={
            "fcnet_hiddens": [64, 64],
            "fcnet_activation": "relu",
        },
        num_sgd_iter=10,
        train_batch_size=4000,
    ).resources(num_gpus=0)

    # Step 3: Training the agent
    trainer = config.build()

    for i in range(100):  # Number of training iterations
        result = trainer.train()
        print(f"Iteration {i}: reward = {result['episode_reward_mean']}")

    # Cleanup
    ray.shutdown()

if __name__ == "__main__":
    main()
