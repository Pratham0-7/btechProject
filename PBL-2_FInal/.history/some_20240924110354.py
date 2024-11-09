import os
import subprocess

# Set paths to SUMO binaries and your map.net.xml file
sumo_tools_path = r"D:\sumo\tools"  # Update this with your SUMO tools path
sumo_bin = r"D:\sumo\bin"  # Update with SUMO bin path
net_file = r"D:\PBL-2_FInal\map.net.xml"  # Correct path to your map.net.xml
route_file = r"D:\PBL-2_FInal\generated_routes.rou.xml"  # Output route file

# Generate random trips file using randomTrips.py (SUMO Tool)
trips_file = route_file.replace('.rou.xml', '.trips.xml')
random_trips_cmd = [
    'python', os.path.join(sumo_tools_path, 'randomTrips.py'),
    '-n', net_file,  # Network file
    '-r', trips_file,  # Output trips file
    '--end', '1000',  # End time for simulation
    '--binomial', '2',  # Number of vehicles added per time step
    '--route-file', route_file  # Optional, directly generate route file
]

# Run the command
try:
    subprocess.run(random_trips_cmd, check=True)
    print("Trip file and route file created successfully.")
except subprocess.CalledProcessError as e:
    print(f"Error generating trips and routes: {e}")

# Generate routes from trips using DUAROUTER
duarouter_cmd = [
    os.path.join(sumo_bin, 'duarouter'),
    '-n', net_file,  # Network file
    '-t', trips_file,  # Input trips file
    '-o', route_file,  # Output route file
    '--ignore-errors',  # Ignore minor errors
]

# Run the command
try:
    subprocess.run(duarouter_cmd, check=True)
    print(f"Route file created: {route_file}")
except subprocess.CalledProcessError as e:
    print(f"Error generating route file: {e}")
