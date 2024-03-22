import numpy as np
import cv2
import matplotlib.pyplot as plt


image_path = r'C:\Users\Jonathan van Zyl\Documents\GitHub\paparazzi\COMPUTERVISIONSIM\Projection\single_image\image\68016093.jpg'
time_stamp = 68.016093

#image_path = r'C:\Users\Jonathan van Zyl\Documents\GitHub\paparazzi\COMPUTERVISIONSIM\Projection\single_image\image\63582791.jpg'
#time_stamp = 63.582791


# Extract data from the csv
csv_path = r'C:\Users\Jonathan van Zyl\Documents\GitHub\paparazzi\COMPUTERVISIONSIM\Projection\single_image\20190121-140303.csv'

# Extract time using pandas
import pandas as pd
df = pd.read_csv(csv_path)
time = df['time']
pos_x = df['pos_x']
pos_y = df['pos_y']
pos_z = df['pos_z']
att_phi = df['att_phi']
att_theta = df['att_theta']
att_psi = df['att_psi']

# Interpolate the data with respect to the time stamp
index = np.argmin(np.abs(time - time_stamp))

# Extract the position and attitude at the time stamp
pos_x = pos_x[index]
pos_y = pos_y[index]
pos_z = pos_z[index]
att_phi = att_phi[index]
att_theta = att_theta[index]
att_psi = att_psi[index]

# make state vector
state_vector = np.array([pos_x, pos_y, pos_z, att_phi, att_theta, att_psi])

# Rotation matrix wrt to cyberzoo frame to drone frame
def update_camera_rotation_matrix(phi, psi, theta):
            # Converting from [-pi, pi] to [0, 2pi]:
            if phi < 0:
                roll = 2 * np.pi + phi
            else:
                roll = phi
            if psi < 0:
                yaw = 2 * np.pi + psi
            else:
                roll = phi
                yaw = psi
            pitch = theta

            Rx = np.array([[1, 0, 0],
                   [0, np.cos(roll), -np.sin(roll)],
                   [0, np.sin(roll), np.cos(roll)]])

            Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                        [0, 1, 0],
                        [-np.sin(pitch), 0, np.cos(pitch)]])

            Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                        [np.sin(yaw), np.cos(yaw), 0],
                        [0, 0, 1]])

            # The order of multiplication depends on the convention used. This uses ZYX (yaw-pitch-roll).
            R = np.dot(Rz, np.dot(Ry, Rx))

            return R

'''
_OZ1: X=-1.34 m, Y=-3.37 m, Z=0 m
_OZ2: X=-4.40 m, Y=1.05 m, Z=0 m
_OZ3: X=0.44 m, Y=4.18 m, Z=0 m
_OZ4: X=3.27 m, Y=-0.41 m, Z=0 m
'''

# Define the corners of the cyberzoo in Cyberzoo frame
OZ1 = np.array([-1.34, -3.37, 0])
OZ2 = np.array([-4.40, 1.05, 0])
OZ3 = np.array([0.44, 4.18, 0])
OZ4 = np.array([3.27, -0.41, 0])

# Rotate & translate the corners into the drone frame
R = update_camera_rotation_matrix(att_phi, att_psi, att_theta)
t = np.array([pos_x, pos_y, pos_z])
OZ1 = np.dot(R, OZ1 - t)
OZ2 = np.dot(R, OZ2 - t)
OZ3 = np.dot(R, OZ3 - t)
OZ4 = np.dot(R, OZ4 - t)

'''
X_camera = Y_drone
            Y_camera = Z_drone
            Z_camera = X_drone
'''

# Define the corners of the cyberzoo in the drone frame
OZ1 = np.array([OZ1[1], OZ1[2], OZ1[0]])
OZ2 = np.array([OZ2[1], OZ2[2], OZ2[0]])
OZ3 = np.array([OZ3[1], OZ3[2], OZ3[0]])
OZ4 = np.array([OZ4[1], OZ4[2], OZ4[0]])
               

# Define the camera and distortion matrix
f_x = 300.
f_y = 300.
c_x = 350.
c_y = 200.
K = np.array([[f_x,   0.,         c_x],
            [  0.,         f_y, c_y],
            [  0.,           0.,           1.        ]])


D = np.array([[ 0.],
            [ 0. ],
            [0.],
            [ 0.]])

# Ensure the points are in the correct format
OZ1 = np.array([OZ1], dtype=np.float32).reshape(-1, 1, 3)
OZ2 = np.array([OZ2], dtype=np.float32).reshape(-1, 1, 3)
OZ3 = np.array([OZ3], dtype=np.float32).reshape(-1, 1, 3)
OZ4 = np.array([OZ4], dtype=np.float32).reshape(-1, 1, 3)

rvec_null = np.zeros((1, 1, 3), dtype=np.float32)
Tvec_null = np.zeros((1, 1, 3), dtype=np.float32)



# Fisheye project the corners
OZ1 = cv2.fisheye.projectPoints(OZ1, rvec_null, Tvec_null, K, D)
OZ2 = cv2.fisheye.projectPoints(OZ2, rvec_null, Tvec_null, K, D)
OZ3 = cv2.fisheye.projectPoints(OZ3, rvec_null, Tvec_null, K, D)
OZ4 = cv2.fisheye.projectPoints(OZ4, rvec_null, Tvec_null, K, D)

# Define the image
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# Rotate image
image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
# Draw the corners on the image
cv2.circle(image, (int(OZ1[0][0][0][0]), int(OZ1[0][0][0][1])), 5, (0, 255, 0), -1)
cv2.circle(image, (int(OZ2[0][0][0][0]), int(OZ2[0][0][0][1])), 5, (0, 255, 0), -1)
cv2.circle(image, (int(OZ3[0][0][0][0]), int(OZ3[0][0][0][1])), 5, (0, 255, 0), -1)
cv2.circle(image, (int(OZ4[0][0][0][0]), int(OZ4[0][0][0][1])), 5, (0, 255, 0), -1)

# Draw the interpolated points
# Interpolate points between the corners
num_points = 20
x_values = np.linspace(OZ1[0][0][0][0], OZ2[0][0][0][0], num_points)
y_values = np.linspace(OZ1[0][0][0][1], OZ2[0][0][0][1], num_points)
for i in range(num_points):
    cv2.circle(image, (int(x_values[i]), int(y_values[i])), 5, (0, 0, 0), -1) # Black : (0, 0, 0)

num_points = 20
x_values = np.linspace(OZ2[0][0][0][0], OZ3[0][0][0][0], num_points)
y_values = np.linspace(OZ2[0][0][0][1], OZ3[0][0][0][1], num_points)
for i in range(num_points):
    cv2.circle(image, (int(x_values[i]), int(y_values[i])), 5, (0, 0, 225), -1) # Red : (0, 0, 255)

num_points = 20
x_values = np.linspace(OZ3[0][0][0][0], OZ4[0][0][0][0], num_points)
y_values = np.linspace(OZ3[0][0][0][1], OZ4[0][0][0][1], num_points)
for i in range(num_points):
    cv2.circle(image, (int(x_values[i]), int(y_values[i])), 5, (255, 0, 0), -1) # Blue : (255, 0, 0)

num_points = 20
x_values = np.linspace(OZ4[0][0][0][0], OZ1[0][0][0][0], num_points)
y_values = np.linspace(OZ4[0][0][0][1], OZ1[0][0][0][1], num_points)
for i in range(num_points):
    cv2.circle(image, (int(x_values[i]), int(y_values[i])), 5, (0, 165, 255), -1)  # Orange : (0, 165, 255)

# Display the image
plt.imshow(image)
plt.show()

# Use a clicker to find pixel coordinates of the corners
# Clicker
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)

#cv2.imshow('image', image)
#cv2.setMouseCallback('image', click_event)
#cv2.waitKey(0)

# 248 160
        

import cv2
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import interact

# Function to project points and update the image based on camera parameters
def update_image(f_x, f_y, c_x, c_y):
    K = np.array([[f_x, 0., c_x],
                  [0., f_y, c_y],
                  [0., 0., 1.]])
    D = np.zeros((4, 1))  # Assuming zero distortion for simplicity

    # Project the corner points using the updated camera matrix
    OZ1_projected = cv2.fisheye.projectPoints(OZ1, rvec_null, Tvec_null, K, D)
    OZ2_projected = cv2.fisheye.projectPoints(OZ2, rvec_null, Tvec_null, K, D)
    OZ3_projected = cv2.fisheye.projectPoints(OZ3, rvec_null, Tvec_null, K, D)
    OZ4_projected = cv2.fisheye.projectPoints(OZ4, rvec_null, Tvec_null, K, D)

    # Copy the original image so as not to overwrite it
    updated_image = np.copy(image)

    # Draw the projected points and interpolated points on the image
    # Similar to the drawing part of your original code but using the `updated_image`
    # ...

    # Display the updated image
    plt.imshow(updated_image)
    plt.show()

# Define sliders for the camera parameters
f_x_slider = widgets.FloatSlider(value=300.0, min=200.0, max=600.0, step=10.0, description='f_x:')
f_y_slider = widgets.FloatSlider(value=300.0, min=200.0, max=600.0, step=10.0, description='f_y:')
c_x_slider = widgets.FloatSlider(value=350.0, min=200.0, max=500.0, step=10.0, description='c_x:')
c_y_slider = widgets.FloatSlider(value=200.0, min=100.0, max=400.0, step=10.0, description='c_y:')

# Use interact to create the interactive widget
interact(update_image, f_x=f_x_slider, f_y=f_y_slider, c_x=c_x_slider, c_y=c_y_slider)
