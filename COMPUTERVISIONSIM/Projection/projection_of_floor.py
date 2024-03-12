import numpy as np
import cv2
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter, FuncAnimation

plot_bool = False
image_bool = True
animation_bool = True

# Use the Camera class from projection_functions.py
from projection_functions import Camera, StateVector, CyberZooStructure, VideoFeed

# Data paths (Adjust to your own directory path)
file_path = "C:/Users/aname/Documents/GitHub/paparazzi/AE4317_2019_datasets/AE4317_2019_datasets/cyberzoo_poles_panels/20190121-140303.csv"
image_path = "C:/Users/aname/Documents/GitHub/paparazzi/AE4317_2019_datasets/AE4317_2019_datasets/cyberzoo_poles_panels/20190121-140205"
image_path = r'C:/Users/aname/Documents/GitHub/paparazzi/AE4317_2019_datasets/AE4317_2019_datasets/cyberzoo_poles_panels/20190121-140205'

image_path = r"C:\Users\Jonathan van Zyl\Documents\GitHub\paparazzi\Data_gitignore\AE4317_2019_datasets\cyberzoo_poles_panels\20190121-140205"
file_path = "Data_gitignore/AE4317_2019_datasets/cyberzoo_poles_panels/20190121-140303.csv"

image_path = 'COMPUTERVISIONSIM/Projection/single_image/image'
file_path = 'COMPUTERVISIONSIM/Projection/single_image/20190121-140303.csv'

# Create a camera object
camera_front = Camera()

# Create a state vector object
state_vector = StateVector(file_path)
# Extract Cyberzoo data
zmin = max(state_vector.z_pos_array)#max(state_vector.z_pos_array)
cyberzoo = CyberZooStructure(zmin)
points3d_cyberzoo = cyberzoo.return_points3d()
#perimeterspoints_3dWorld_Cyberzoo = cyberzoo.get_perimeter_points()
formatted_colored_points = cyberzoo.get_colored_perimeter_points()

# Load the image
images = VideoFeed(image_path)
# Rotate the image
images.image_rotate_90_counter()


# Loop over the images
x_pos_camera = []
y_pos_camera = []
z_pos_camera = []
theta_camera = []
phi_camera = []
psi_camera = []
times_list = []

for i in range(len(images.frame_files)):
    images.index = i
    images.image_current = images.image_read(i)
    images.image_rotate_90_counter()
    
    time = images.find_time()
    times_list.append(time)
    # update the camera object
    camera_front.update_state_vector(state_vector, time)
    x_pos_camera.append(camera_front.x_pos); y_pos_camera.append(camera_front.y_pos); z_pos_camera.append(camera_front.z_pos)
    theta_camera.append(camera_front.theta); phi_camera.append(camera_front.phi); psi_camera.append(camera_front.psi)


    # Create the projection
    # Project the cyberzoo points
    # Convert points3d_cyberzoo elements to a numpy array of type float32 and reshape to have 3 channels
    points2D_cyberzoo_XYRGB, points3D_cyberzoo_camera_XYZRBG = camera_front.project_3D_to_2D(np.array(formatted_colored_points, dtype=np.float32).reshape(-1, 1, 6), fisheye_bool = True)
    images.draw_circle(points2D_cyberzoo_XYRGB,radius=10)

    print('theta_camera', theta_camera)
    print('phi_camera', phi_camera)
    print('psi_camera', psi_camera)
    print('x_pos_camera', x_pos_camera)
    print('y_pos_camera', y_pos_camera)
    print('z_pos_camera', z_pos_camera)


    # Display the image
    if image_bool:
        images.image_show(waitKeyvalue = 1000000)



if plot_bool:
    from Plotter import full_plotter
    full_plotter(times_list, x_pos_camera, y_pos_camera, z_pos_camera, state_vector)


# ANIMATOR A WIP, doesn't work with current code and needs to be changed