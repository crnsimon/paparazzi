import numpy as np
import cv2
import pandas as pd

# Use the Camera class from projection_functions.py
from projection_functions import Camera, StateVector, CyberZooStructure, VideoFeed

# Create a camera object
camera = Camera(0, 0, 0, 0, 0, 0)

# Create a state vector object
file_path = "Data_gitignore/AE4317_2019_datasets/cyberzoo_poles_panels/20190121-140303.csv"
state_vector = StateVector(file_path)

# Extract Cyberzoo data
zmin = state_vector.z_pos_array[0]
cyberzoo = CyberZooStructure(zmin)

# Load the image
image_path = "/home/jonathan/paparazzi/Data_gitignore/AE4317_2019_datasets/cyberzoo_poles_panels/20190121-140205"
images = VideoFeed(image_path)
# Rotate the image
images.image_rotate_90_counter()
# Show the image
#images.image_show()


# Apply a filter
images.filter()


# Resize the image
images.resize_frame()

# Optical flow
frame1 = images.image_read(0)
frame2 = images.image_read(1)
flowclass = OpticalFlow(frame1, frame2)