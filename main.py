import numpy as np
import traci
import time

GRAPH = {
    "node_1": {"node_2": 1, "node_3": 4},
    "node_2": {"node_1": 1, "node_3": 2, "node_4": 5},
    "node_3": {"node_1": 4, "node_2": 2, "node_4": 1},
    "node_4": {"node_2": 5, "node_3": 1}
}

NODES = {
    "node_1": (0, 0),
    "node_2": (1, 0),
    "node_3": (0, 1),
    "node_4": (1, 1)
}

CHARGING_STATIONS = ["node_4"]

def euclidean_heuristic(node, goal):
    pos_node = np.array(NODES[node])
    pos_goal = np.array(NODES[goal])
    return np.linalg.norm(pos_node - pos_goal)

def astar(start, goal, graph, heuristic):
    open_set = [(0, start)]
    came_from = {}
    g_score = {node: float("inf") for node in graph}
    g_score[start] = 0
    f_score = {node: float("inf") for node in graph}
    f_score[start] = heuristic(start, goal)
    
    while open_set:
        _, current = min(open_set, key=lambda x: x[0])
        open_set = [i for i in open_set if i[1] != current]

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            return path[::-1]

        for neighbor, cost in graph[current].items():
            tentative_g_score = g_score[current] + cost
            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                open_set.append((f_score[neighbor], neighbor))

    return []

class SumoEnv:
    def __init__(self, sumo_cmd):
        self.sumo_cmd = sumo_cmd
        self.done = False

    def reset(self):
        if traci.isLoaded():
            traci.close()

        traci.start(self.sumo_cmd)
        traci.simulationStep()

        # Remove all existing vehicles at the start of each reset
        for vehicle_id in traci.vehicle.getIDList():
            try:
                traci.vehicle.remove(vehicle_id)
            except traci.exceptions.TraCIException as e:
                print(f"Error removing vehicle {vehicle_id}: {e}")

        # Adding unique vehicle IDs for each episode
        self.vehicle_ids = [f"av_{i}" for i in range(3)]
        for vehicle_id in self.vehicle_ids:
            try:
                traci.vehicle.add(vehicle_id, "main_route", typeID="av_type", depart="now")
            except traci.exceptions.TraCIException as e:
                print(f"Error adding vehicle {vehicle_id}: {e}")

        self.done = False
        return self.get_state()

    def get_state(self):
        state = []
        for vehicle_id in traci.vehicle.getIDList():
            position = traci.vehicle.getPosition(vehicle_id)
            distances = [np.linalg.norm(np.array(position) - np.array(NODES[station])) for station in CHARGING_STATIONS]
            state.extend(distances)
        return np.array(state).flatten()

    def step(self, actions):
        rewards = []
        print(f"Actions: {actions}")

        for i, vehicle_id in enumerate(traci.vehicle.getIDList()):
            if actions[i] == 1 and not self.is_at_charging_station(vehicle_id):
                self.move_to_closest_node(vehicle_id)

            position = traci.vehicle.getPosition(vehicle_id)
            distances_to_stations = [np.linalg.norm(np.array(position) - np.array(NODES[station])) for station in CHARGING_STATIONS]

            if distances_to_stations:
                min_distance = min(distances_to_stations)
                rewards.append(-min_distance)
            else:
                rewards.append(-100)

            closest_station_index = np.argmin(distances_to_stations)
            if distances_to_stations[closest_station_index] < 10:
                traci.vehicle.setSpeed(vehicle_id, 0)

        self.done = traci.simulation.getMinExpectedNumber() == 0
        traci.simulationStep()

        if not rewards:
            raise ValueError("Rewards list is empty. Check the step logic.")

        return self.get_state(), rewards, self.done

    def move_to_closest_node(self, vehicle_id):
        current_pos = traci.vehicle.getRoadID(vehicle_id)
        closest_station = min(CHARGING_STATIONS, key=lambda station: np.linalg.norm(np.array(NODES[current_pos]) - np.array(NODES[station])))
        path = astar(current_pos, closest_station, GRAPH, euclidean_heuristic)
        
        if path:
            traci.vehicle.setRoute(vehicle_id, path)

    def is_at_charging_station(self, vehicle_id):
        current_pos = traci.vehicle.getRoadID(vehicle_id)
        return current_pos in CHARGING_STATIONS

if __name__ == "__main__":
    sumo_cmd = ["sumo-gui", "-c", "simulation.sumocfg"]
    env = SumoEnv(sumo_cmd)
    
    num_episodes = 10
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = np.random.choice([0, 1], size=len(traci.vehicle.getIDList()))
            next_state, rewards, done = env.step(action)
            total_reward += sum(rewards)
            state = next_state
        
        print(f"Episode {episode + 1} finished with total reward: {total_reward}")
