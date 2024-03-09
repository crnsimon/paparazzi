import numpy as np
import cv2
import pandas as pd

# Use the Camera class from projection_functions.py
from projection_functions import Camera, StateVector, CyberZooStructure, VideoFeed

# Data paths
file_path = "Data_gitignore/AE4317_2019_datasets/cyberzoo_poles_panels/20190121-140303.csv"
image_path = "/home/jonathan/paparazzi/Data_gitignore/AE4317_2019_datasets/cyberzoo_poles_panels/20190121-140205"
image_path = r'C:\Users\Jonathan van Zyl\Documents\GitHub\paparazzi\Data_gitignore\AE4317_2019_datasets\cyberzoo_poles_panels\20190121-140205'

#image_path = r"C:\Users\Jonathan van Zyl\Documents\GitHub\paparazzi\Data_gitignore\AE4317_2019_datasets\cyberzoo_canvas_approach\20190121-151448"
#file_path = "Data_gitignore/AE4317_2019_datasets/cyberzoo_canvas_approach/20190121-151518.csv"

# Create a camera object
camera_front = Camera()

# Create a state vector object
state_vector = StateVector(file_path)

# Extract Cyberzoo data
zmin = min(state_vector.z_pos_array)
cyberzoo = CyberZooStructure(zmin)
points3d_cyberzoo = cyberzoo.return_points3d()

# Load the image
images = VideoFeed(image_path)

# Rotate the image
images.image_rotate_90_counter()


# Loop over the images
x_pos_camera = []
y_pos_camera = []
z_pos_camera = []
points2d_cyberzoo_0_array = []
points2d_cyberzoo_1_array = []
points2d_cyberzoo_2_array = []
points2d_cyberzoo_3_array = []

for i in range(len(images.frame_files)):
    images.index = i
    images.image_current = images.image_read(i)
    images.image_rotate_90_counter()
    # Add your image processing code here
    # Find and print time
    time = images.find_time()
    # update the camera object
    camera_front.update_state_vector(state_vector, time)
    x_pos_camera.append(camera_front.x_pos)
    y_pos_camera.append(camera_front.y_pos)
    z_pos_camera.append(camera_front.z_pos)

    # Create the projection
    # Project the cyberzoo points
    # Convert points3d_cyberzoo elements to a numpy array of type float32 and reshape to have 3 channels
    points2d_cyberzoo_0 = camera_front.project_3D_to_2D(np.array(points3d_cyberzoo[0], dtype=np.float32).reshape(-1, 1, 3))
    points2d_cyberzoo_1 = camera_front.project_3D_to_2D(np.array(points3d_cyberzoo[1], dtype=np.float32).reshape(-1, 1, 3))
    points2d_cyberzoo_2 = camera_front.project_3D_to_2D(np.array(points3d_cyberzoo[2], dtype=np.float32).reshape(-1, 1, 3))
    points2d_cyberzoo_3 = camera_front.project_3D_to_2D(np.array(points3d_cyberzoo[3], dtype=np.float32).reshape(-1, 1, 3))
    # Draw the points
    images.draw_circle(points2d_cyberzoo_0[0][0], points2d_cyberzoo_0[0][1], radius=10, color=(0, 255, 0))
    images.draw_circle(points2d_cyberzoo_1[0][0], points2d_cyberzoo_1[0][1], radius=10, color =(0,0, 255) ) #Red
    images.draw_circle(points2d_cyberzoo_2[0][0], points2d_cyberzoo_2[0][1], radius=10, color = (255, 0, 0)) #Blue
     #Yellow
    images.draw_circle(points2d_cyberzoo_3[0][0], points2d_cyberzoo_3[0][1], radius=10, color = (0, 255, 255) )
    # Append the points to the array
    points2d_cyberzoo_0_array.append(points2d_cyberzoo_0[0])
    points2d_cyberzoo_1_array.append(points2d_cyberzoo_1[0])
    points2d_cyberzoo_2_array.append(points2d_cyberzoo_2[0])
    points2d_cyberzoo_3_array.append(points2d_cyberzoo_3[0])

    # Display the image
    images.image_show(waitKeyvalue = 100)

# print points3d_cyberzoo
print(points3d_cyberzoo[0])
print(points3d_cyberzoo[1])
print(points3d_cyberzoo[2])
print(points3d_cyberzoo[3])

# Plot the camera position
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x_pos_camera, y_pos_camera, z_pos_camera)
ax.plot(points3d_cyberzoo[0][0], points3d_cyberzoo[0][1], points3d_cyberzoo[0][2], 'ro')
ax.plot(points3d_cyberzoo[1][0], points3d_cyberzoo[1][1], points3d_cyberzoo[1][2], 'go')
ax.plot(points3d_cyberzoo[2][0], points3d_cyberzoo[2][1], points3d_cyberzoo[2][2], 'mo')
ax.plot(points3d_cyberzoo[3][0], points3d_cyberzoo[3][1], points3d_cyberzoo[3][2], 'co')
# Initial camera position
ax.plot(x_pos_camera[0], y_pos_camera[0], z_pos_camera[0], 'go')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

'''
Plot fails verification (i.e.) cyberzoo coordinates in the wrong place
'''

# Plot points2d_cyberzoo
# Convert to numpy array
points2d_cyberzoo_0_array = np.array(points2d_cyberzoo_0_array)
points2d_cyberzoo_1_array = np.array(points2d_cyberzoo_1_array)
points2d_cyberzoo_2_array = np.array(points2d_cyberzoo_2_array)
points2d_cyberzoo_3_array = np.array(points2d_cyberzoo_3_array)
# Plot
fig, ax = plt.subplots()
ax.plot(points2d_cyberzoo_0_array[:,0], points2d_cyberzoo_0_array[:,1], 'ro')
ax.plot(points2d_cyberzoo_1_array[:,0], points2d_cyberzoo_1_array[:,1], 'go')
ax.plot(points2d_cyberzoo_2_array[:,0], points2d_cyberzoo_2_array[:,1], 'mo')
ax.plot(points2d_cyberzoo_3_array[:,0], points2d_cyberzoo_3_array[:,1], 'co')
ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.show()


# Plot state vectors
state_vector.plot_xyz_3d()
state_vector.plot_angles()