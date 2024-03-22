import csv
import numpy as np
import matplotlib.pyplot as plt

'''
csv like this
time	pos_x	pos_y	pos_z	vel_x	vel_y	vel_z	att_phi	att_theta	att_psi	rate_p	rate_q	rate_r	rpm_obs_1
231.277771	-4.457031	0.886719	0	-0.019999	-0.009998	-0.016397	0.006715	0.01471	1.078207	0.003174	0.000244	0.001953	0
231.278748	-4.457031	0.886719	0	-0.019999	-0.009998	-0.016829	0.006715	0.01471	1.078207	-0.002686	-0.00293	-0.003418	0
231.279724	-4.457031	0.886719	0	-0.019999	-0.009998	-0.016941	0.006715	0.01471	1.078207	0.003662	-0.000244	0.002441	0
231.281677	-4.457031	0.886719	0	-0.019999	-0.009998	-0.017006	0.006715	0.01471	1.078207	-0.001465	0.002441	-0.003418	0

'''

import pandas as pd

# Read csv
filepath = r"C:\Users\Jonathan van Zyl\Documents\GitHub\paparazzi\Data_gitignore\20240322-112204.csv"
df = pd.read_csv(filepath)

# Extract 'time', 'pos_x', 'pos_y', and 'pos_z'
data = df[['time', 'pos_x', 'pos_y', 'pos_z', 'att_phi', 'att_theta', 'att_psi']]

# Plot XY graph
plt.figure()
plt.plot(data['pos_x'], data['pos_y'])
plt.xlabel('X position')
plt.ylabel('Y position')
plt.title('XY position')
plt.grid()
plt.show()


# XYZ vector
xyz = np.array([data['pos_x'], data['pos_y'], data['pos_z']])

yaw = -15 * np.pi / 180
Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                        [np.sin(yaw), np.cos(yaw), 0],
                        [0, 0, 1]])

xyz_rot = np.dot(Rz, xyz)

# Plot XY graph
plt.figure()
plt.plot(xyz_rot[0], xyz_rot[1], 'o')
plt.xlabel('X position')
plt.ylabel('Y position')
plt.title('XY position')
# Make Aspect ratio = 1
plt.gca().set_aspect('equal', adjustable='box')
plt.grid()
plt.show()


# How to extract position for each index
#print(data['pos_x'][0])
plot_bool = 0
if plot_bool:
    # Plot with time
    import matplotlib.pyplot as plt

    # Subplot of x_pos y_pos and z_pos
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(data['time'], data['pos_x'])
    plt.title('Position x')
    plt.xlabel('Time [s]')
    plt.ylabel('Position [m]')
    plt.grid()

    plt.subplot(3, 1, 2)
    plt.plot(data['time'], data['pos_y'])
    plt.title('Position y')
    plt.xlabel('Time [s]')
    plt.ylabel('Position [m]')
    plt.grid()

    plt.subplot(3, 1, 3)
    plt.plot(data['time'], data['pos_z'])
    plt.title('Position z')
    plt.xlabel('Time [s]')
    plt.ylabel('Position [m]')
    plt.grid()

    plt.tight_layout()
    plt.show()


import numpy as np

# Function to convert lat/lon to meters from a reference point
def latlon_to_xyz(lat, lon, home_lat, home_lon):
    # Earth's radius in meters
    R = 6371000
    
    # Convert degrees to radians
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    home_lat_rad = np.radians(home_lat)
    home_lon_rad = np.radians(home_lon)
    
    # Approximate meters per degree
    meters_per_lat = 111000
    meters_per_lon = 111000 * np.cos(home_lat_rad)
    
    # Delta calculation
    delta_lat = lat - home_lat
    delta_lon = lon - home_lon
    
    # Convert delta degrees to meters
    x = delta_lon * meters_per_lon
    y = delta_lat * meters_per_lat
    z = 0  # Assuming constant altitude
    
    return x, y, z

# HOME coordinates
home_lat = 51.990631
home_lon = 4.376796

# OZ waypoints
oz_waypoints = [
    {"name": "_OZ1", "lat": 51.9906006, "lon": 4.3767764},
    {"name": "_OZ2", "lat": 51.9906405, "lon": 4.3767316},
    {"name": "_OZ3", "lat": 51.9906687, "lon": 4.3768025},
    {"name": "_OZ4", "lat": 51.9906273, "lon": 4.3768438}
]

# Convert each OZ waypoint to XYZ
for oz in oz_waypoints:
    x, y, z = latlon_to_xyz(oz["lat"], oz["lon"], home_lat, home_lon)
    print(f"{oz['name']}: X={x:.2f} m, Y={y:.2f} m, Z={z} m")

# Rotate the XYZ coordinates
yaw = -15 * np.pi / 180

Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                [np.sin(yaw), np.cos(yaw), 0],
                [0, 0, 1]])

# Convert each OZ waypoint to XYZ

for oz in oz_waypoints:
    x, y, z = latlon_to_xyz(oz["lat"], oz["lon"], home_lat, home_lon)
    xyz = np.array([x, y, z])
    xyz_rot = np.dot(Rz, xyz)
    print(f"{oz['name']}: X={xyz_rot[0]:.2f} m, Y={xyz_rot[1]:.2f} m, Z={xyz_rot[2]:.2f} m")

# Plot the OZ waypoints
plt.figure()
for oz in oz_waypoints:
    x, y, z = latlon_to_xyz(oz["lat"], oz["lon"], home_lat, home_lon)
    plt.plot(x, y, 'o', label=oz["name"])
plt.xlabel('X position [m]')
plt.ylabel('Y position [m]')
plt.title('OZ waypoints')
plt.grid()
plt.legend()
plt.show()

# Plot the rotated OZ waypoints
plt.figure()
for oz in oz_waypoints:
    x, y, z = latlon_to_xyz(oz["lat"], oz["lon"], home_lat, home_lon)
    xyz = np.array([x, y, z])
    xyz_rot = np.dot(Rz, xyz)
    plt.plot(xyz_rot[0], xyz_rot[1], 'o', label=oz["name"])
plt.xlabel('X position [m]')
plt.ylabel('Y position [m]')
plt.title('OZ waypoints rotated')
plt.grid()
plt.show()