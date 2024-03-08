import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2

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


# Cyberzoo dimensions
cyberzoo_shapes = {}
cyberzoo_shapes['cyberzoo_green_width'] = 7 #m
cyberzoo_shapes['cyberzoo_green_length'] = 7 #m
cyberzoo_shapes['z_min'] = 0#m - set to original value
# Cyberzoo corner coordinates
cyberzoo_shapes['corner_coordinates'] = {
    'A': {'x' : cyberzoo_shapes['cyberzoo_green_width']/2,
          'y' : cyberzoo_shapes['cyberzoo_green_length']/2,
          'z' : cyberzoo_shapes['z_min']},
    'B': {'x' : -cyberzoo_shapes['cyberzoo_green_width']/2,
          'y' : cyberzoo_shapes['cyberzoo_green_length']/2,
          'z' : cyberzoo_shapes['z_min']},
    'C': {'x' : -cyberzoo_shapes['cyberzoo_green_width']/2,
          'y' : -cyberzoo_shapes['cyberzoo_green_length']/2,
          'z' : cyberzoo_shapes['z_min']},
    'D': {'x' : cyberzoo_shapes['cyberzoo_green_width']/2,
          'y' : -cyberzoo_shapes['cyberzoo_green_length']/2,
          'z' : cyberzoo_shapes['z_min']}
}

cyberzoo_shapes['wall_height'] = 4 #m
cyberzoo_shapes['corner_upper_coordinates'] = {
    'A': {'x' : cyberzoo_shapes['cyberzoo_green_width']/2,
          'y' : cyberzoo_shapes['cyberzoo_green_length']/2,
          'z' : cyberzoo_shapes['z_min'] + cyberzoo_shapes['wall_height']},
    'B': {'x' : -cyberzoo_shapes['cyberzoo_green_width']/2,
          'y' : cyberzoo_shapes['cyberzoo_green_length']/2,
          'z' : cyberzoo_shapes['z_min'] + cyberzoo_shapes['wall_height']},
    'C': {'x' : -cyberzoo_shapes['cyberzoo_green_width']/2,
          'y' : -cyberzoo_shapes['cyberzoo_green_length']/2,
          'z' : cyberzoo_shapes['z_min'] + cyberzoo_shapes['wall_height']},
    'D': {'x' : cyberzoo_shapes['cyberzoo_green_width']/2,
          'y' : -cyberzoo_shapes['cyberzoo_green_length']/2,
          'z' : cyberzoo_shapes['z_min'] + cyberzoo_shapes['wall_height']}
}


# FLOOR
# Extract the corner coordinates
corners = cyberzoo_shapes['corner_coordinates']
# Create arrays for x, y, and z coordinates
x = np.array([corners[corner]['x'] for corner in corners])
y = np.array([corners[corner]['y'] for corner in corners])
z = np.array([corners[corner]['z'] for corner in corners])
# Create a meshgrid for the surface
X, Y = np.meshgrid(x, y)
Z = np.full_like(X, cyberzoo_shapes['z_min'])


# Left wall : A, D, A_upper, D_upper
corners_upper = cyberzoo_shapes['corner_upper_coordinates']
x_left_wall = np.array([corners['A']['x'], corners['D']['x'], corners_upper['A']['x'], corners_upper['D']['x']])
y_left_wall = np.array([corners['A']['y'], corners['D']['y'], corners_upper['A']['y'], corners_upper['D']['y']])
z_left_wall = np.array([corners['A']['z'], corners['D']['z'], corners_upper['A']['z'], corners_upper['D']['z']])
X_left, Y_left = np.meshgrid(x_left_wall, y_left_wall)
Z_left = np.full_like(X_left, z_left_wall)


# Right wall : B, C, B_upper, C_upper
x_right_wall = np.array([corners['B']['x'], corners['C']['x'], corners_upper['B']['x'], corners_upper['C']['x']])
y_right_wall = np.array([corners['B']['y'], corners['C']['y'], corners_upper['B']['y'], corners_upper['C']['y']])
z_right_wall = np.array([corners['B']['z'], corners['C']['z'], corners_upper['B']['z'], corners_upper['C']['z']])
X_right, Y_right = np.meshgrid(x_right_wall, y_right_wall)
Z_right = np.full_like(X_right, z_right_wall)

# Back wall : C, D, C_upper, D_upper
x_back_wall = np.linspace(corners['C']['x'], corners['D']['x'], num=100)
y_back_wall = np.linspace(corners['C']['y'], corners['D']['y'], num=100)
z_back_wall = np.linspace(corners['C']['z'], corners_upper['C']['z'], num=100)
X_back, Z_back = np.meshgrid(x_back_wall, z_back_wall)
Y_back = np.full_like(X_back, y_back_wall.mean())

# Front wall : A, B, A_upper, B_upper
x_front_wall = np.linspace(corners['A']['x'], corners['B']['x'], num=100)
y_front_wall = np.linspace(corners['A']['y'], corners['B']['y'], num=100)
z_front_wall = np.linspace(corners['A']['z'], corners_upper['A']['z'], num=100)
X_front, Z_front = np.meshgrid(x_front_wall, z_front_wall)
Y_front = np.full_like(X_front, y_front_wall.mean())



# Plot the data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot a square for the coordinates of the square and fill it in
ax.plot_surface(X, Y, Z, color='b')

# Plot the left wall
ax.plot_surface(X_left, Y_left, Z_left, color='r')

# Plot the right wall
ax.plot_surface(X_right, Y_right, Z_right, color='c')

# Plot the back wall
ax.plot_surface(X_back, Y_back, Z_back, color='g')

# Plot the front wall
ax.plot_surface(X_front, Y_front, Z_front, color='y')

ax.plot(x_array, y_array, z_array)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()



frames_dir = "Data_gitignore/AE4317_2019_datasets/cyberzoo_poles_panels"
#frames_dir = "Data_gitignore/AE4317_2019_datasets/cyberzoo_poles_panels_mats_bottomcam"

for j in os.listdir(frames_dir):
    folder = os.path.join(frames_dir, j)

    if os.path.isdir(folder):
        # Get a list of image file names sorted in ascending order
        frame_files = sorted(os.listdir(folder))
        for i in range(0, len(frame_files)-1):
            #  STEP 1: read & rotate the frames
            # Read the image
            img = cv2.imread(os.path.join(folder, frame_files[i]))

            # Rotate the image
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

            cv2.imshow('image', img)
            