import os
import sys
import traci
import gym
import numpy as np
from gym import spaces
import random
from lxml import etree
import subprocess

# Function to preprocess the XML
def preprocess_xml(input_file, output_file):
    tree = etree.parse(input_file)
    root = tree.getroot()
    
    for edge in root.findall('.//edge'):
        if edge.get('id') == 'edge_to_remove':  # Customize as needed
            root.remove(edge)
    
    tree.write(output_file, pretty_print=True, xml_declaration=True, encoding='UTF-8')

# Function to generate routes
def generate_routes(net_file, route_file):
    command = ["duarouter", "--net-file", net_file, "--route-files", route_file]
    subprocess.run(command)


# Path to SUMO tools
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
        
        if action == 0:  # Forward
            traci.vehicle.setSpeed(vehicle_id, 10)
        elif action == 1:  # Left
            traci.vehicle.changeLane(vehicle_id, 0, 10)
        elif action == 2:  # Right
            traci.vehicle.changeLane(vehicle_id, 1, 10)
        elif action == 3:  # Stop
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

def q_learning(env, episodes=1000, alpha=0.1, gamma=0.99, epsilon=0.1):
    q_table = np.zeros((100, 4))
    
    for episode in range(episodes):
        state = env.reset()
        done = False
        
        while not done:
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()  # Explore
            else:
                action = np.argmax(q_table[int(state[1])])  # Exploit
            
            next_state, reward, done, _ = env.step(action)
            old_value = q_table[int(state[1]), action]
            next_max = np.max(q_table[int(next_state[1])])
            q_table[int(state[1]), action] = old_value + alpha * (reward + gamma * next_max - old_value)
            
            state = next_state
        
        if episode % 100 == 0:
            print(f"Episode {episode} completed")
    
    return q_table

if __name__ == "__main__":
    input_net_file = "D:/PBL-2_FInal/map.net.xml"
    output_net_file = "D:/PBL-2_FInal/map_cleaned.net.xml"
    preprocess_xml(input_net_file, output_net_file)
    
    route_file = "D:/PBL-2_FInal/routes.xml"
    generate_routes(output_net_file, route_file, num_vehicles=100)  # Generate routes based on cleaned net file
    
    sumo_cfg_file = "D:/PBL-2_FInal/map.sumocfg"
    charging_stations = [(100, 100), (200, 200)]  # Replace with actual coordinates
    
    env = SumoEnv(sumo_cfg_file, output_net_file, charging_stations)
    
    q_table = q_learning(env, episodes=1000)
    print("Training completed")
    
    np.save("q_table.npy", q_table)
