import os
import sys
import traci
import gym
import numpy as np
from gym import spaces

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
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=100, shape=(2,), dtype=np.float32)
        self.sumo_process = None
        self.sumo_cmd = ["sumo", "-c", sumo_cfg_file]
        
    def reset(self):
        if self.sumo_process is not None:
            traci.close()
        self.sumo_process = traci.start(self.sumo_cmd)
        traci.simulationStep()
        self.battery_capacity = 100
        vehicle_id = "vehicle_0"
        traci.vehicle.add(vehicle_id, routeID="route0")
        traci.vehicle.setMaxSpeed(vehicle_id, 10)
        return self.get_observation(vehicle_id)
    
    def step(self, action):
        vehicle_id = "vehicle_0"
        if action == 0:
            traci.vehicle.setSpeed(vehicle_id, 10)
        elif action == 1:
            traci.vehicle.changeLane(vehicle_id, 0, 10)
        elif action == 2:
            traci.vehicle.changeLane(vehicle_id, 1, 10)
        elif action == 3:
            traci.vehicle.setSpeed(vehicle_id, 0)
        traci.simulationStep()
        self.battery_capacity -= 1
        observation = self.get_observation(vehicle_id)
        reward = self.calculate_reward(vehicle_id)
        done = self.battery_capacity <= 0 or self.reached_charging_station(vehicle_id)
        return observation, reward, done, {}
    
    def get_observation(self, vehicle_id):
        position = traci.vehicle.getPosition(vehicle_id)
        return np.array([position[0], self.battery_capacity], dtype=np.float32)
    
    def calculate_reward(self, vehicle_id):
        vehicle_position = traci.vehicle.getPosition(vehicle_id)
        closest_station_dist = min(np.linalg.norm(np.array(vehicle_position) - np.array(station))
                                   for station in self.charging_stations)
        reward = -closest_station_dist
        if self.battery_capacity < 10:
            reward -= 10
        return reward
    
    def reached_charging_station(self, vehicle_id):
        vehicle_position = traci.vehicle.getPosition(vehicle_id)
        for station in self.charging_stations:
            if np.linalg.norm(np.array(vehicle_position) - np.array(station)) < 10:
                return True
        return False
    
    def close(self):
        traci.close()
