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
animation_bool = False

# Use the Camera class from projection_functions.py
from projection_functions import Camera, StateVector, CyberZooStructure, VideoFeed, add_color_to_points, blend_colors

# Data paths (Adjust to your own directory path)
file_path = "C:/Users/aname/Documents/GitHub/paparazzi/AE4317_2019_datasets/AE4317_2019_datasets/cyberzoo_poles_panels/20190121-140303.csv"
image_path = "C:/Users/aname/Documents/GitHub/paparazzi/AE4317_2019_datasets/AE4317_2019_datasets/cyberzoo_poles_panels/20190121-140205"
image_path = r'C:/Users/aname/Documents/GitHub/paparazzi/AE4317_2019_datasets/AE4317_2019_datasets/cyberzoo_poles_panels/20190121-140205'

image_path = r"C:\Users\Jonathan van Zyl\Documents\GitHub\paparazzi\Data_gitignore\AE4317_2019_datasets\cyberzoo_poles_panels\20190121-140205"
file_path = "Data_gitignore/AE4317_2019_datasets/cyberzoo_poles_panels/20190121-140303.csv"

# Create a camera object
camera_front = Camera()

# Create a state vector object
state_vector = StateVector(file_path)
# Extract Cyberzoo data
zmin = max(state_vector.z_pos_array)#max(state_vector.z_pos_array)
cyberzoo = CyberZooStructure(zmin)
points3d_cyberzoo = cyberzoo.return_points3d()
perimeterspoints_3dWorld_Cyberzoo = cyberzoo.get_perimeter_points()

# Define the RGB color codes for each corner
corner_colors = {
    'A': (0, 255, 0),  # Green
    'B': (255, 0, 0),  # Red
    'C': (0, 0, 255),  # Blue
    'D': (255, 255, 0)  # Yellow
}

# Get colored points for each line
AB_colored_points = add_color_to_points(cyberzoo.generate_line_points(cyberzoo.corner_coordinates['A'], cyberzoo.corner_coordinates['B']), corner_colors['A'], corner_colors['B'])
BC_colored_points = add_color_to_points(cyberzoo.generate_line_points(cyberzoo.corner_coordinates['B'], cyberzoo.corner_coordinates['C'])[1:], corner_colors['B'], corner_colors['C'])
CD_colored_points = add_color_to_points(cyberzoo.generate_line_points(cyberzoo.corner_coordinates['C'], cyberzoo.corner_coordinates['D'])[1:], corner_colors['C'], corner_colors['D'])
DA_colored_points = add_color_to_points(cyberzoo.generate_line_points(cyberzoo.corner_coordinates['D'], cyberzoo.corner_coordinates['A'])[1:], corner_colors['D'], corner_colors['A'])

# Combine all colored points
colored_perimeter_points = AB_colored_points + BC_colored_points + CD_colored_points + DA_colored_points

# Convert to a more friendly format for display
formatted_colored_points = np.array([(*point, *color) for point, color in colored_perimeter_points]) # (X, Y, Z, R, G, B)


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
    # Add your image processing code here
    # Find and print time
    time = images.find_time()
    times_list.append(time)
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
    points2D_cyberzoo_XYRGB, points3D_cyberzoo_camera_XYZRBG = camera_front.project_3D_to_2D(np.array(formatted_colored_points, dtype=np.float32).reshape(-1, 1, 6))
    images.draw_circle(points2D_cyberzoo_XYRGB,radius=70)

    # Display the image
    if image_bool:
        images.image_show(waitKeyvalue = 100)

times = np.array(times_list)
sorted_indices = np.argsort(times)
times_sorted = times[sorted_indices]

# Plot x, y and z coordinates of red corner (green in the mp4) in camera frame
if plot_bool:
    corner_colors = ["Red", "Green", "Magenta", "Cyan"]
    corner_coordinates = [points3d_cyberzoo_0_camera_array, points3d_cyberzoo_1_camera_array,
                          points3d_cyberzoo_2_camera_array, points3d_cyberzoo_3_camera_array]
    for corner_nr in range(4):

        x_values_list = [point[0] for point in corner_coordinates[corner_nr]]
        y_values_list = [point[1] for point in corner_coordinates[corner_nr]]
        z_values_list = [point[2] for point in corner_coordinates[corner_nr]]
        x_values, y_values, z_values = np.array(x_values_list), np.array(y_values_list), np.array(z_values_list)
        x_sorted = x_values[sorted_indices]
        y_sorted = y_values[sorted_indices]
        z_sorted = z_values[sorted_indices]

        fig, axs = plt.subplots(3)
        fig.suptitle(f'{corner_colors[corner_nr]} Corner Coordinates in Camera Frame')
        axs[0].plot(times_sorted, x_sorted)
        axs[0].set_title('X')
        axs[1].plot(times_sorted, y_sorted)
        axs[1].set_title('Y')
        axs[2].plot(times_sorted, z_sorted)
        axs[2].set_title('Z')
        plt.show()

if plot_bool:
    x_pos, y_pos, z_pos = np.array(x_pos_camera), np.array(y_pos_camera), np.array(z_pos_camera)
    x_pos_sorted = x_pos[sorted_indices]
    y_pos_sorted = y_pos[sorted_indices]
    z_pos_sorted = z_pos[sorted_indices]

    fig, axs = plt.subplots(3)
    fig.suptitle('Drone Coordinates in Cyberzoo Frame')
    axs[0].plot(times_sorted, x_pos_sorted)
    axs[0].set_title('X')
    axs[1].plot(times_sorted, y_pos_sorted)
    axs[1].set_title('Y')
    axs[2].plot(times_sorted, z_pos_sorted)
    axs[2].set_title('Z')
    plt.show()




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
    ax.plot(x_pos_camera[0], y_pos_camera[0], z_pos_camera[0], 'bo')
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
    # Conversion from [-pi, pi] to [0, 2pi]
    if phi < 0:
        phi = 2 * np.pi + phi
    if psi < 0:
        psi = 2 * np.pi + psi
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
    # Create a new figure
    fig = plt.figure()

    # Create a 3D axis
    ax = fig.add_subplot(111, projection='3d')

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
    ax.invert_yaxis()
    ax.invert_zaxis()

    # Setting the axes properties
    # ax.set_xlim3d([min(x_pos_camera), max(x_pos_camera)])
    # ax.set_ylim3d([min(y_pos_camera), max(y_pos_camera)])
    # ax.set_zlim3d([min(z_pos_camera), max(z_pos_camera)])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Define the update function for the animation
    # Create the animation
    # Sort all values w.r.t time
    x_pos_array = np.array(x_pos_camera)
    y_pos_array = np.array(y_pos_camera)
    z_pos_array = np.array(z_pos_camera)
    theta_pos_array = np.array(theta_camera)
    phi_pos_array = np.array(phi_camera)
    psi_pos_array = np.array(psi_camera)

    x_pos_sorted = x_pos_array[sorted_indices]
    y_pos_sorted = y_pos_array[sorted_indices]
    z_pos_sorted = z_pos_array[sorted_indices]
    theta_pos_sorted = theta_pos_array[sorted_indices]
    phi_pos_sorted = phi_pos_array[sorted_indices]
    psi_pos_sorted = psi_pos_array[sorted_indices]

    # Highlight fourth wall
    ax.xaxis.set_pane_color((0, 1.0, 0, 1.0))

    # ani = FuncAnimation(fig, update_camera_orientation, len(x_pos_camera),
    #                     fargs=(x_pos_camera, y_pos_camera, z_pos_camera, theta_camera, phi_camera, psi_camera, lines),
    #                     interval=100, blit=False)

    ani = FuncAnimation(fig, update_camera_orientation, len(x_pos_camera),
                        fargs=(x_pos_sorted, y_pos_sorted, z_pos_sorted, theta_pos_sorted, phi_pos_sorted, psi_pos_sorted, lines),
                        interval=100, blit=False)

    # Save the animation
    writer = FFMpegWriter(fps=15, metadata=dict(artist='Jonny'), bitrate=1800)
    ani.save("CameraCoordinateSystem.mp4", writer=writer)

    # Display the plot
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
   