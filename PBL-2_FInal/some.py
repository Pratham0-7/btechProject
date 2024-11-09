import gym
from gym import spaces
import numpy as np
import traci
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.algorithms.sac import SAC
import ray
from ray import tune

class MultiAgentSumoEnv(MultiAgentEnv):
    def __init__(self, config):
        super(MultiAgentSumoEnv, self).__init__()

        # Define action and observation space for each agent
        self.num_agents = config["num_agents"]  # Number of agents
        self.action_space = spaces.Discrete(5)  # Example: 5 discrete actions per agent
        self.observation_space = spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)

        # Initialize SUMO environment
        self._initialize_sumo()

    def _initialize_sumo(self):
        sumoBinary = "sumo"  # Use "sumo-gui" if you want to visualize the simulation
        sumoCmd = ["sumo", "-c", "path/to/your/map.sumocfg"]  # Specify SUMO config path
        traci.start(sumoCmd)

    def reset(self):
        # Reset SUMO and get the initial observations for each agent
        traci.close()  # Close any existing connection
        self._initialize_sumo()  # Restart the SUMO process
        observations = {f"agent_{i}": self._get_initial_observation() for i in range(self.num_agents)}
        return observations

    def step(self, action_dict):
        # Apply actions for each agent
        for agent_id, action in action_dict.items():
            self._apply_action(agent_id, action)

        # Simulate one step in SUMO
        traci.simulationStep()

        # Collect observations, rewards, done flags, and info for each agent
        observations = {f"agent_{i}": self._get_observation() for i in range(self.num_agents)}
        rewards = {f"agent_{i}": self._calculate_reward(i) for i in range(self.num_agents)}
        dones = {f"agent_{i}": self._check_done(i) for i in range(self.num_agents)}
        dones["__all__"] = all(dones.values())  # Episode ends if all agents are done
        infos = {f"agent_{i}": {} for i in range(self.num_agents)}

        return observations, rewards, dones, infos

    def _apply_action(self, agent_id, action):
        # Define how each agent's action affects the SUMO environment
        pass  # Implement action application here

    def _get_observation(self):
        # Get current observations from SUMO for an agent
        return np.random.rand(10)  # Replace with actual SUMO observation data

    def _calculate_reward(self, agent_id):
        # Calculate reward for the agent based on the environment state
        return 1.0  # Replace with actual reward logic

    def _check_done(self, agent_id):
        # Determine if the episode is done for the agent
        return traci.simulation.getMinExpectedNumber() <= 0

    def _get_initial_observation(self):
        # Get the initial state for an agent when resetting the environment
        return self._get_observation()

    def close(self):
        # Close the SUMO simulation
        traci.close()

# Set up RLlib environment configuration
env_config = {
    "num_agents": 5  # Set the number of agents
}

# Register the environment with RLlib
ray.init(ignore_reinit_error=True)
tune.register_env("multi_agent_sumo_env", lambda config: MultiAgentSumoEnv(config))

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.sac import SACConfig

# Define configurations for PPO and SAC
ppo_config = PPOConfig().environment(
    env="multi_agent_sumo_env", env_config=env_config
).rollouts(num_rollout_workers=1).training(
    model={"fcnet_hiddens": [256, 256]},
    train_batch_size=4000,
    sgd_minibatch_size=128,
    num_sgd_iter=30,
)

sac_config = SACConfig().environment(
    env="multi_agent_sumo_env", env_config=env_config
).rollouts(num_rollout_workers=1).training(
    model={"fcnet_hiddens": [256, 256]},
    train_batch_size=4000,
    sgd_minibatch_size=128,
    num_sgd_iter=30,
)
# Instantiate PPO and SAC algorithms
ppo_trainer = PPO(config=ppo_config)
sac_trainer = SAC(config=sac_config)

# Training loop
num_iterations = 100
for i in range(num_iterations):
    ppo_result = ppo_trainer.train()
    sac_result = sac_trainer.train()
    
    print(f"Iteration {i}: PPO Reward = {ppo_result['episode_reward_mean']}, SAC Reward = {sac_result['episode_reward_mean']}")

    # Save checkpoints every 10 iterations
    if i % 10 == 0:
        ppo_checkpoint = ppo_trainer.save("path/to/ppo_checkpoint")
        sac_checkpoint = sac_trainer.save("path/to/sac_checkpoint")
        print(f"Saved PPO checkpoint at {ppo_checkpoint}")
        print(f"Saved SAC checkpoint at {sac_checkpoint}")
import matplotlib.pyplot as plt

# Example code to plot rewards over iterations
ppo_rewards = [result["episode_reward_mean"] for result in ppo_results]
sac_rewards = [result["episode_reward_mean"] for result in sac_results]

plt.figure(figsize=(10, 6))
plt.plot(range(num_iterations), ppo_rewards, label="PPO")
plt.plot(range(num_iterations), sac_rewards, label="SAC")
plt.xlabel("Iterations")
plt.ylabel("Average Reward")
plt.title("PPO vs SAC Rewards Over Time")
plt.legend()
plt.show()
