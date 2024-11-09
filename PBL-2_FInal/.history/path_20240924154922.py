def predict_vehicle_path(self, vehicle_id, steps_ahead=10):
        if vehicle_id not in self.vehicles or len(self.vehicles[vehicle_id]) < 2:
            return None
        
        (x1, y1), (x2, y2) = self.vehicles[vehicle_id][-2:]

        
        dx = x2 - x1
        dy = y2 - y1

        predicted_path = []
        for i in range(1, steps_ahead + 1):
        
            predicted_x = x2 + i * dx
            predicted_y = y2 + i * dy
            predicted_path.append((predicted_x, predicted_y))

        return predicted_path

def print_predicted_paths(self):
        for vehicle_id in self.vehicles:
            predicted_path = self.predict_vehicle_path(vehicle_id)
            if predicted_path:
                print(f"Predicted path for vehicle {vehicle_id}: {predicted_path}")
