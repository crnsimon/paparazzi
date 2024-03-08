import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt

class Camera:
    def __init__(self):
        self.view_is_uptodate = False
        self.proj_is_uptodate = False
        self.view_matrix = np.identity(4)
        
        self.fov_y = np.pi / 3
        self.near_dist = 1.0
        self.far_dist = 50000.0
        
        self.vp_x = 0
        self.vp_y = 0
        self.position = np.array([100.0, 100.0, 100.0])
        self.target = np.array([0.0, 0.0, 0.0])
        self.orientation = np.identity(3)

    def set_viewport(self, offsetx, offsety, width, height):
        self.vp_x = offsetx
        self.vp_y = offsety
        self.vp_width = width
        self.vp_height = height
        self.proj_is_uptodate = False

    def set_fovy(self, value):
        self.fov_y = value
        self.proj_is_uptodate = False

    def direction(self):
        return -self.orientation @ np.array([0.0, 0.0, 1.0])

    def up(self):
        return self.orientation @ np.array([0.0, 1.0, 0.0])

    def right(self):
        return self.orientation @ np.array([1.0, 0.0, 0.0])

    def set_direction(self, new_direction):
        up = self.up()
        new_direction = -new_direction / np.linalg.norm(new_direction)
        right = np.cross(up, new_direction)
        right = right / np.linalg.norm(right)
        up = np.cross(new_direction, right)
        self.orientation = np.column_stack((right, up, new_direction))
        self.view_is_uptodate = False

    def set_target(self, target):
        self.target = target
        if not np.allclose(self.target, self.position):
            new_direction = self.target - self.position
            self.set_direction(new_direction / np.linalg.norm(new_direction))

    def set_position(self, position):
        self.position = position
        self.view_is_uptodate = False

    def set_orientation(self, orientation):
        self.orientation = orientation
        self.view_is_uptodate = False

    def update_view_matrix(self):
        if not self.view_is_uptodate:
            q = self.orientation.T
            self.view_matrix[:3, :3] = q
            self.view_matrix[:3, 3] = -q @ self.position
            self.view_is_uptodate = True

    def update_projection_matrix(self):
        if not self.proj_is_uptodate:
            self.projection_matrix = np.identity(4)
            aspect = float(self.vp_width) / self.vp_height
            theta = self.fov_y / 2.0
            range = self.far_dist - self.near_dist
            invtan = 1.0 / np.tan(theta)
            
            self.projection_matrix[0, 0] = invtan / aspect
            self.projection_matrix[1, 1] = invtan
            self.projection_matrix[2, 2] = -(self.near_dist + self.far_dist) / range
            self.projection_matrix[3, 2] = -1.0
            self.projection_matrix[2, 3] = -2.0 * self.near_dist * self.far_dist / range
            self.projection_matrix[3, 3] = 0.0
            
            self.proj_is_uptodate = True

    def get_view_matrix(self):
        self.update_view_matrix()
        return self.view_matrix

    def get_projection_matrix(self):
        self.update_projection_matrix()
        return self.projection_matrix

    # Other methods like localTranslate, zoom, etc., can be added similarly by translating their C++ counterparts.

def project_point(camera, point_3d):
    # Ensure the camera matrices are up to date
    camera.update_view_matrix()
    camera.update_projection_matrix()

    # Convert the 3D point to homogeneous coordinates
    point_3d_homogeneous = np.append(point_3d, 1)
    point_camera = np.dot(camera.get_view_matrix(), point_3d_homogeneous)
    point_clip = np.dot(camera.get_projection_matrix(), point_camera)

    # Perspective division to get normalized device coordinates
    point_ndc = point_clip[:3] / point_clip[3]

    # Transform to image coordinates
    # Assuming the viewport transformation is the simple case where x and y range from 0 to width and height respectively
    point_image_x = (point_ndc[0] + 1) * 0.5 * camera.vp_width
    point_image_y = (1 - point_ndc[1]) * 0.5 * camera.vp_height  # y is inverted

    return point_image_x, point_image_y

def get_target_from_orientation(pitch, yaw):
    # Convert degrees to radians for trigonometric functions
    pitch_rad = np.radians(pitch)
    yaw_rad = np.radians(yaw)

    # Assuming the drone's camera is fixed and always looks straight out from the drone,
    # Calculate the direction vector the camera is pointing in based on pitch and yaw
    x = np.cos(pitch_rad) * np.sin(yaw_rad)
    y = np.sin(pitch_rad)
    z = np.cos(pitch_rad) * np.cos(yaw_rad)

    # This direction vector points from the drone's position to where the camera is looking
    direction = np.array([x, y, z])

    return direction

# Example usage:
camera = Camera()

# Read the CSV file into a DataFrame
file_path = "Data_gitignore/AE4317_2019_datasets/cyberzoo_poles_panels/20190121-140303.csv"
df = pd.read_csv(file_path)

# Convert the DataFrame into a dictionary of arrays
data = df.to_dict('list')

# Now data is a dictionary where the keys are the column names and the values are lists of column values

time_array = data['time']
x_array = data['pos_x']
y_array = data['pos_y']
z_array = np.array(data['pos_z'])
pitch_array = data['att_theta']
yaw_array = data['att_psi']


# FLOOR
# Cyberzoo dimensions
cyberzoo_shapes = {}
cyberzoo_shapes['cyberzoo_green_width'] = 7 #m
cyberzoo_shapes['cyberzoo_green_length'] = 7 #m
cyberzoo_shapes['z_min'] = min(z_array)#m - set to original value
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
# Extract the corner coordinates
corners = cyberzoo_shapes['corner_coordinates']
# Create arrays for x, y, and z coordinates
x = np.array([corners[corner]['x'] for corner in corners])
y = np.array([corners[corner]['y'] for corner in corners])
z = np.array([corners[corner]['z'] for corner in corners])

frames_dir = "Data_gitignore/AE4317_2019_datasets/cyberzoo_poles_panels"
point_2d_a_x_list = []
point_2d_a_y_list = []
point_2d_b_x_list = []
point_2d_b_y_list = []
point_2d_c_x_list = []
point_2d_c_y_list = []
point_2d_d_x_list = []
point_2d_d_y_list = []
for j in os.listdir(frames_dir):
    folder = os.path.join(frames_dir, j)

    if os.path.isdir(folder):
        # Get a list of image file names sorted in ascending order
        frame_files = sorted(os.listdir(folder))

        for t, frame_file in enumerate(frame_files):
            # Set camera parameters based on drone's state
            camera.set_position(np.array([x_array[t], y_array[t], z_array[t]]))  # Example position
            pitch = pitch_array[t]
            yaw = yaw_array[t]
            target = get_target_from_orientation(pitch, yaw)

            camera.set_target(target)  # Camera's target
            
            camera.set_viewport(0, 0, 4096, 3072)  # Example viewport dimensions
            
            point_3d_corner_a = np.array([x[0], y[0], z[0]])  # Example 3D point
            # Project the 3D point onto the 2D image plane
            point_2d_a = project_point(camera, point_3d_corner_a)

            point_3d_corner_b = np.array([x[1], y[1], z[1]])  # Example 3D point
            # Project the 3D point onto the 2D image plane
            point_2d_b = project_point(camera, point_3d_corner_b)

            point_3d_corner_c = np.array([x[2], y[2], z[2]])  # Example 3D point
            # Project the 3D point onto the 2D image plane
            point_2d_c = project_point(camera, point_3d_corner_c)

            point_3d_corner_d = np.array([x[3], y[3], z[3]])  # Example 3D point
            # Project the 3D point onto the 2D image plane
            point_2d_d = project_point(camera, point_3d_corner_d)

            # Load and display the image via cv
            frame1_bgr = cv2.imread(os.path.join(folder, frame_files[t]))
            frame1_bgr = cv2.rotate(frame1_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)

            # Draw the projected points on the image - make circle bigger
            cv2.circle(frame1_bgr, (int(point_2d_a[0]), int(point_2d_a[1])), 100, (0, 0, 255), -1)
            cv2.circle(frame1_bgr, (int(point_2d_b[0]), int(point_2d_b[1])), 100, (0, 0, 255), -1)
            cv2.circle(frame1_bgr, (int(point_2d_c[0]), int(point_2d_c[1])), 100, (0, 0, 255), -1)
            cv2.circle(frame1_bgr, (int(point_2d_d[0]), int(point_2d_d[1])), 100, (0, 0, 255), -1)

            # Append 2d points to arrays
            point_2d_a_x_list.append(point_2d_a[0])
            point_2d_a_y_list.append(point_2d_a[1])
            point_2d_b_x_list.append(point_2d_b[0])
            point_2d_b_y_list.append(point_2d_b[1])
            point_2d_c_x_list.append(point_2d_c[0])
            point_2d_c_y_list.append(point_2d_c[1])
            point_2d_d_x_list.append(point_2d_d[0])
            point_2d_d_y_list.append(point_2d_d[1])

            # Display image - slow it down
            #cv2.imshow('Frame', frame1_bgr)
            #cv2.waitKey(1)  # Delay in milliseconds (e.g. 100ms = 0.1s)
            #print(t)

from mpl_toolkits.mplot3d import Axes3D

index_array = np.arange(len(point_2d_a_x_list))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Assuming point_2d_a_z_list exists and has the same length as the other two lists
ax.scatter(index_array, point_2d_a_x_list, point_2d_a_y_list, c='r', marker='o')

plt.show()


# plot the x y and z
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x_array, y_array, z_array)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()