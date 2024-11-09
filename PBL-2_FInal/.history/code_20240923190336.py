import os
import sys
import traci
import gym
import numpy as np
from gym import spaces
import random

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

class SumoEnv(gym.Env):
    def __init__(self, sumo_cfg_file, net_file, charging_stations):
        super(SumoEnv, self).__init__()
        
        self.sumo_cfg_file = sumo_cfg_file
        self.net_file = net_file
        self.charging_stations = charging_stations  
        self.battery_capacity = 100 
        
        # Define action and observation space
        self.action_space = spaces.Discrete(4)  # 4 possible directions (e.g., forward, left, right, stop)
        self.observation_space = spaces.Box(low=0, high=100, shape=(2,), dtype=np.float32)  # position and battery
        
        # SUMO setup
        self.sumo_process = None
        self.sumo_cmd = ["sumo", "-c", sumo_cfg_file]
        
    def reset(self):
        if self.sumo_process is not None:
            traci.close()
        self.sumo_process = traci.start(self.sumo_cmd)
        traci.simulationStep()
        
        # Initialize battery life and vehicle's position
        self.battery_capacity = 100
        vehicle_id = "vehicle_0"
        traci.vehicle.add(vehicle_id, routeID="route0")
        traci.vehicle.setMaxSpeed(vehicle_id, 10)
        
        return self.get_observation(vehicle_id)
    
    def step(self, action):
        vehicle_id = "vehicle_0"
        
        # Take action (e.g., move vehicle)
        if action == 0:  # Forward
            traci.vehicle.setSpeed(vehicle_id, 10)
        elif action == 1:  # Left
            traci.vehicle.changeLane(vehicle_id, 0, 10)  # Simplified
        elif action == 2:  # Right
            traci.vehicle.changeLane(vehicle_id, 1, 10)  # Simplified
        elif action == 3:  # Stop
            traci.vehicle.setSpeed(vehicle_id, 0)
        
        # Move simulation forward
        traci.simulationStep()
        
        # Reduce battery life as vehicle moves
        self.battery_capacity -= 1
        
        # Get new observation
        observation = self.get_observation(vehicle_id)
        
        # Calculate reward (e.g., negative reward if battery is low, positive if reached charging station)
        reward = self.calculate_reward(vehicle_id)
        
        # Check if the episode is done (battery depleted or charging station reached)
        done = self.battery_capacity <= 0 or self.reached_charging_station(vehicle_id)
        
        return observation, reward, done, {}
    
    def get_observation(self, vehicle_id):
        # Get the vehicle's position and battery level
        position = traci.vehicle.getPosition(vehicle_id)
        return np.array([position[0], self.battery_capacity], dtype=np.float32)
    
    def calculate_reward(self, vehicle_id):
        # Example reward: more reward for being closer to a charging station, penalty for low battery
        vehicle_position = traci.vehicle.getPosition(vehicle_id)
        closest_station_dist = min(np.linalg.norm(np.array(vehicle_position) - np.array(station))
                                   for station in self.charging_stations)
        
        reward = -closest_station_dist  # Negative reward proportional to distance to the charging station
        if self.battery_capacity < 10:
            reward -= 10  # Penalize for low battery
        
        return reward
    
    def reached_charging_station(self, vehicle_id):
        vehicle_position = traci.vehicle.getPosition(vehicle_id)
        for station in self.charging_stations:
            if np.linalg.norm(np.array(vehicle_position) - np.array(station)) < 10:  # Reached if within 10 meters
                return True
        return False
    
    def close(self):
        traci.close()
