import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
plot_bool = False
image_bool = False
animation_bool = True

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
theta_camera = []
phi_camera = []
psi_camera = []

points2d_cyberzoo_0_array = []
points2d_cyberzoo_1_array = []
points2d_cyberzoo_2_array = []
points2d_cyberzoo_3_array = []

points3d_cyberzoo_0_camera_array = []
points3d_cyberzoo_1_camera_array = []
points3d_cyberzoo_2_camera_array = []
points3d_cyberzoo_3_camera_array = []

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
    theta_camera.append(camera_front.theta)
    phi_camera.append(camera_front.phi)
    psi_camera.append(camera_front.psi)

    # Create the projection
    # Project the cyberzoo points
    # Convert points3d_cyberzoo elements to a numpy array of type float32 and reshape to have 3 channels
    points2d_cyberzoo_0, points3d_cyberzoo_0_camera = camera_front.project_3D_to_2D(np.array(points3d_cyberzoo[0], dtype=np.float32).reshape(-1, 1, 3))
    points2d_cyberzoo_1, points3d_cyberzoo_1_camera = camera_front.project_3D_to_2D(np.array(points3d_cyberzoo[1], dtype=np.float32).reshape(-1, 1, 3))
    points2d_cyberzoo_2, points3d_cyberzoo_2_camera = camera_front.project_3D_to_2D(np.array(points3d_cyberzoo[2], dtype=np.float32).reshape(-1, 1, 3))
    points2d_cyberzoo_3, points3d_cyberzoo_3_camera = camera_front.project_3D_to_2D(np.array(points3d_cyberzoo[3], dtype=np.float32).reshape(-1, 1, 3))
    '''
    points2d_cyberzoo_0 = camera_front.project_3D_to_2D_non_fisheye(np.array(points3d_cyberzoo[0], dtype=np.float32).reshape(-1, 1, 3))
    points2d_cyberzoo_1 = camera_front.project_3D_to_2D_non_fisheye(np.array(points3d_cyberzoo[1], dtype=np.float32).reshape(-1, 1, 3))
    points2d_cyberzoo_2 = camera_front.project_3D_to_2D_non_fisheye(np.array(points3d_cyberzoo[2], dtype=np.float32).reshape(-1, 1, 3))
    points2d_cyberzoo_3 = camera_front.project_3D_to_2D_non_fisheye(np.array(points3d_cyberzoo[3], dtype=np.float32).reshape(-1, 1, 3))
    '''
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
   
    # Append the points to the array
    points3d_cyberzoo_0_camera_array.append(points3d_cyberzoo_0_camera[0][0])
    points3d_cyberzoo_1_camera_array.append(points3d_cyberzoo_1_camera[0][0])
    points3d_cyberzoo_2_camera_array.append(points3d_cyberzoo_2_camera[0][0])
    points3d_cyberzoo_3_camera_array.append(points3d_cyberzoo_3_camera[0][0])

    # Display the image
    if image_bool:
        images.image_show(waitKeyvalue = 1)


# Plot the camera position
if plot_bool:
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
if plot_bool:
    fig, ax = plt.subplots()
    ax.plot(points2d_cyberzoo_0_array[:,0], points2d_cyberzoo_0_array[:,1], 'ro')
    ax.plot(points2d_cyberzoo_1_array[:,0], points2d_cyberzoo_1_array[:,1], 'go')
    ax.plot(points2d_cyberzoo_2_array[:,0], points2d_cyberzoo_2_array[:,1], 'mo')
    ax.plot(points2d_cyberzoo_3_array[:,0], points2d_cyberzoo_3_array[:,1], 'co')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.show()


# Plot state vectors
if plot_bool:
    state_vector.plot_xyz_3d()
    state_vector.plot_angles()


'''

ANIMATION OF THE CAMERA'S REFERENCE FRAME

'''

def update_camera_orientation(num, x_pos_camera, y_pos_camera, z_pos_camera, theta_camera, phi_camera, psi_camera, lines):
    # Update the data for the camera's position
    lines[0].set_data(np.array([x_pos_camera[num]]), np.array([y_pos_camera[num]]))
    lines[0].set_3d_properties(np.array([z_pos_camera[num]]))

    # Calculate the camera's orientation vectors
    R = calculate_rotation_matrix(theta_camera[num], phi_camera[num], psi_camera[num])
    
    # Orientation vector length
    vector_length = 0.5

    # Orientation vectors
    x_vector = R @ np.array([vector_length, 0, 0])
    y_vector = R @ np.array([0, vector_length, 0])
    z_vector = R @ np.array([0, 0, vector_length])

    # Update the orientation lines
    for i, vec in enumerate([x_vector, y_vector, z_vector]):
        lines[i + 1].set_data(np.array([x_pos_camera[num], x_pos_camera[num] + vec[0]]),
                             np.array([y_pos_camera[num], y_pos_camera[num] + vec[1]]))
        lines[i + 1].set_3d_properties(np.array([z_pos_camera[num], z_pos_camera[num] + vec[2]]))

    return lines


def calculate_rotation_matrix(theta, phi, psi):
    # Assuming theta is pitch, phi is roll, and psi is yaw
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(phi), -np.sin(phi)],
                   [0, np.sin(phi), np.cos(phi)]])
    Ry = np.array([[np.cos(theta), 0, np.sin(theta)],
                   [0, 1, 0],
                   [-np.sin(theta), 0, np.cos(theta)]])
    Rz = np.array([[np.cos(psi), -np.sin(psi), 0],
                   [np.sin(psi), np.cos(psi), 0],
                   [0, 0, 1]])
    R = Rz @ Ry @ Rx
    return R

# Create a figure and a 3D axis
if animation_bool:
    fig = plt.figure()
    ax = p3.Axes3D(fig)

    # Initial position and orientation lines for the camera
    line0 = ax.plot([x_pos_camera[0]], [y_pos_camera[0]], [z_pos_camera[0]], 'bo')[0]
    lines = [line0]

    # Orientation lines (X - red, Y - green, Z - blue)
    colors = ['r', 'g', 'b']
    for color in colors:
        lines.append(ax.plot([0, 0], [0, 0], [0, 0], color)[0])

    # Plot the cyberzoo corners
    ax.plot(points3d_cyberzoo[0][0], points3d_cyberzoo[0][1], points3d_cyberzoo[0][2], 'go')
    ax.plot(points3d_cyberzoo[1][0], points3d_cyberzoo[1][1], points3d_cyberzoo[1][2], 'ro')
    ax.plot(points3d_cyberzoo[2][0], points3d_cyberzoo[2][1], points3d_cyberzoo[2][2], 'bo')
    ax.plot(points3d_cyberzoo[3][0], points3d_cyberzoo[3][1], points3d_cyberzoo[3][2], 'yo')

    # Setting the axes properties
    #ax.set_xlim3d([min(x_pos_camera), max(x_pos_camera)])
    #ax.set_ylim3d([min(y_pos_camera), max(y_pos_camera)])
    #ax.set_zlim3d([min(z_pos_camera), max(z_pos_camera)])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Create the animation
    ani = animation.FuncAnimation(fig, update_camera_orientation, len(x_pos_camera), fargs=(x_pos_camera, y_pos_camera, z_pos_camera, theta_camera, phi_camera, psi_camera, lines), interval=100, blit=False)
    # Save the animation
    writer = FFMpegWriter(fps=15, metadata=dict(artist='Jonny'), bitrate=1800)
    ani.save("CameraCoordinateSystem.mp4", writer=writer)
    plt.show()


# Plot the points3d_cyberzoo_0_camera_array eg points3d_cyberzoo_0_camera_array
# Convert to numpy array
points3d_cyberzoo_0_camera_array = np.array(points3d_cyberzoo_0_camera_array)
points3d_cyberzoo_1_camera_array = np.array(points3d_cyberzoo_1_camera_array)
points3d_cyberzoo_2_camera_array = np.array(points3d_cyberzoo_2_camera_array)
points3d_cyberzoo_3_camera_array = np.array(points3d_cyberzoo_3_camera_array)
# Animation of the points3d_cyberzoo_0_camera_array
if animation_bool:

    # Initialize the figure and axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Initialize the plots
    line0, = ax.plot([], [], [], 'ro')
    line1, = ax.plot([], [], [], 'go')
    line2, = ax.plot([], [], [], 'mo')
    line3, = ax.plot([], [], [], 'co')

    # Set the limits of the plot
    ax.set_xlim([min(points3d_cyberzoo_0_camera_array[:,0]), max(points3d_cyberzoo_0_camera_array[:,0])])
    ax.set_ylim([min(points3d_cyberzoo_0_camera_array[:,1]), max(points3d_cyberzoo_0_camera_array[:,1])])
    ax.set_zlim([min(points3d_cyberzoo_0_camera_array[:,2]), max(points3d_cyberzoo_0_camera_array[:,2])])

    # Update function for the animation
    def update(num):
        line0.set_data(points3d_cyberzoo_0_camera_array[:num, 0], points3d_cyberzoo_0_camera_array[:num, 1])
        line0.set_3d_properties(points3d_cyberzoo_0_camera_array[:num, 2])
        line1.set_data(points3d_cyberzoo_1_camera_array[:num, 0], points3d_cyberzoo_1_camera_array[:num, 1])
        line1.set_3d_properties(points3d_cyberzoo_1_camera_array[:num, 2])
        line2.set_data(points3d_cyberzoo_2_camera_array[:num, 0], points3d_cyberzoo_2_camera_array[:num, 1])
        line2.set_3d_properties(points3d_cyberzoo_2_camera_array[:num, 2])
        line3.set_data(points3d_cyberzoo_3_camera_array[:num, 0], points3d_cyberzoo_3_camera_array[:num, 1])
        line3.set_3d_properties(points3d_cyberzoo_3_camera_array[:num, 2])
        return line0, line1, line2, line3,

    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=range(len(points3d_cyberzoo_0_camera_array)), blit=True)
    # Save the animation
    writer = FFMpegWriter(fps=15, metadata=dict(artist='Jonny'), bitrate=1800)
    ani.save("CyberzooCornersInCameraCoordinateFrame.mp4", writer=writer)
    # Show the animation
    plt.show()