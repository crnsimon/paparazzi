import numpy as np
import cv2
import pandas as pd

# Use the Camera class from projection_functions.py
from projection_functions import Camera, StateVector, CyberZooStructure, VideoFeed, OpticalFlow

# Create a camera object
camera_front = Camera(0, 0, 0, 0, 0, 0)

# Create a state vector object
file_path = "Data_gitignore/AE4317_2019_datasets/cyberzoo_poles_panels/20190121-140303.csv"
state_vector = StateVector(file_path)

# Extract Cyberzoo data
zmin = state_vector.z_pos_array[0]
cyberzoo = CyberZooStructure(zmin)
points3d_cyberzoo = cyberzoo.return_points3d()

# Load the image
image_path = "/home/jonathan/paparazzi/Data_gitignore/AE4317_2019_datasets/cyberzoo_poles_panels/20190121-140205"
image_path = r'C:\Users\Jonathan van Zyl\Documents\GitHub\paparazzi\Data_gitignore\AE4317_2019_datasets\cyberzoo_poles_panels\20190121-140205'
images = VideoFeed(image_path)
# Rotate the image
images.image_rotate_90_counter()

points2d_cyberzoo = []
# Loop over data set
for i in range(state_vector.number_of_rows()):
    camera_front.update_state_vector(state_vector, i)
    for j in range(len(points3d_cyberzoo)):
        points3d = np.array(points3d_cyberzoo[j], dtype=np.float32).reshape(-1, 1, 3)
        points2d_cyberzoo = camera_front.project_3D_to_2D(points3d)
        points2d_cyberzoo = np.array(points2d_cyberzoo)

'''
TBD Check the frequency of the image and state and align the updates of the camera and state vector, iirc was somewhere in the slides.
'''
# Count number of jpgs
jpgs = images.number_of_images()
print(jpgs)
