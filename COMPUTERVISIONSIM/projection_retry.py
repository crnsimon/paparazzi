import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2

def project_points_fisheye(points_3D, K, D, R, T):
    """
    Project 3D points onto a 2D fisheye camera image plane.

    :param points_3D: Nx3 numpy array of 3D points in the world coordinate system.
    :param K: 3x3 camera matrix.
    :param D: Distortion coefficients for the fisheye lens. A 4x1 or 1x4 array.
    :param R: 3x3 Rotation matrix representing the camera orientation.
    :param T: Translation vector representing the camera position.
    :return: 2D projections of the 3D points as Nx2 numpy array.
    """
    # Ensure points_3D is a homogeneous coordinate matrix
    points_3D_hom = np.hstack([points_3D, np.ones((points_3D.shape[0], 1))])

    # Apply rotation and translation (extrinsic parameters)
    cam_points_3D = np.dot(points_3D_hom, np.hstack([R, T]).T)

    # Project points using fisheye model
    points_2D = cv2.fisheye.projectPoints(cam_points_3D.reshape(-1, 1, 3), np.eye(3), np.zeros((3, 1)), K, D)

    # Extract x, y coordinates from points_2D
    points_2D = points_2D[0].reshape(-1, 2)

    return points_2D

def euler_to_rotation_matrix(roll, pitch, yaw):
    """
    Convert Euler angles to a rotation matrix.

    :param roll: Rotation around the x-axis in radians.
    :param pitch: Rotation around the y-axis in radians.
    :param yaw: Rotation around the z-axis in radians.
    :return: 3x3 rotation matrix.
    """
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

def find_R_and_T(position, orientation):
    """
    Compute the rotation matrix and translation vector from vehicle's state.

    :param position: The vehicle's position as a tuple or list (x, y, z) in global coordinates.
    :param orientation: The vehicle's orientation as Euler angles (roll, pitch, yaw) in radians.
    :return: A tuple containing the rotation matrix R and translation vector T.
    """
    # Unpack the position and orientation
    x, y, z = position
    roll, pitch, yaw = orientation

    # Compute the rotation matrix from Euler angles
    R = euler_to_rotation_matrix(roll, pitch, yaw)

    # The translation vector is simply the position
    T = np.array([[x], [y], [z]])

    return R, T


# Read the CSV file into a DataFrame
file_path = "Data_gitignore/AE4317_2019_datasets/cyberzoo_poles_panels/20190121-140303.csv"
df = pd.read_csv(file_path)

# Convert the DataFrame into a dictionary of arrays
data = df.to_dict('list')

# Now data is a dictionary where the keys are the column names and the values are lists of column values

time_array = data['time']
x_array = data['pos_x']
y_array = data['pos_y']
z_array = -np.array(data['pos_z'])
pitch_array = data['att_theta']
yaw_array = data['att_psi']
roll_array = data['att_phi']
'''
# Generate the camera matrix
mtf9f002_zoom = 1
mtf9f002_offset_x = 0
mtf9f002_offset_y = 0
mtf9f002_output_width = 1920 # Could be different
mtf9f002_output_height = 1080 # Could be different

mtf9f002_focal_x = (mtf9f002_zoom * mtf9f002_output_width / 2)
mtf9f002_focal_y = (mtf9f002_zoom * mtf9f002_output_height / 2)
mtf9f002_center_x = (mtf9f002_output_width * (.5 - mtf9f002_zoom * mtf9f002_offset_x))
mtf9f002_center_y = (mtf9f002_output_height * (.5 - mtf9f002_zoom * mtf9f002_offset_y))

K = np.array([[mtf9f002_focal_x, 0, mtf9f002_center_x],
                  [0, mtf9f002_focal_y, mtf9f002_center_y],
                  [0, 0, 1]])
D = np.array([1.25, 0, 0, 0, 0])
'''

K = np.array([[589.98363697,   0,         117.18359156],
 [  0,         600.54137529, 261.48275908],
 [  0,           0,           1        ]])
D = np.array([[-0.32043809,  0.27653614, -0.06730844, -0.04503392, -2.50539621]])


# Generate R and T for each point in the arrays
R_array = []
T_array = []
for i in range(len(x_array)):
    R, T = find_R_and_T((x_array[i], y_array[i], z_array[i]), (roll_array[i], pitch_array[i], yaw_array[i]))
    R_array.append(R)
    T_array.append(T)


    # Generate 3D points
    points_3D = np.array([[0, 0, 0],
                        [0, 0, 1],
                        [0, 1, 0],
                        [0, 1, 1],
                        [1, 0, 0],
                        [1, 0, 1],
                        [1, 1, 0],
                        [1, 1, 1]])

    # Project the 3D points onto the image plane
    points_2D = project_points_fisheye(points_3D, K, D, R, T)