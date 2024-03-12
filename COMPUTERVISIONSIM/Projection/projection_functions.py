'''
A class for state: x_pos, y_pos, z_pos, x_rot, y_rot, z_rot
where it is the position and rotation of the camera
Has attributes including the camera matrix, the projection matrix, and the view matrix
'''

import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

class Camera:
    def __init__(self):
        self.x_pos = 0
        self.y_pos = 0
        self.z_pos = 0
        self.theta = 0
        self.psi = 0
        self.phi = 0
        
        '''
        self.K_nonfisheye = np.array([[589.98363697,   0,         117.18359156],
                           [  0,         600.54137529, 261.48275908],
                           [  0,           0,           1        ]])
        
        self.D_nonfisheye = np.array([[-0.32043809,  0.27653614, -0.06730844, -0.04503392, -2.50539621]])
        
        
        self.K = np.array([[324.25570292,   0,          25.65423155],
                   [  0,         323.60053988, 265.75527519],
                   [  0,           0,           1.        ]])

        self.D = np.array([[-0.02808937, -0.04655074, 0.0786952, -0.05046657]])
        '''
        # N.B. The nonfisheye take 8,6 on checkborad better to do 9,6 : but my laptop is not powerful enough
        self.K_nonfisheye = np.array([[1.06848861e+03, 0.00000000e+00, 2.43338041e+02],
                              [0.00000000e+00, 1.54122474e+03, 1.22209857e+02],
                              [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

        self.D_nonfisheye = np.array([[7.49344678e-01, -5.89979880e+01, 1.67460462e-01, -7.29040201e-02, 4.51276257e+02]])
        
        self.K = np.array([[323.94986777, 0, 265.6212057 ],
                           [ 0, 324.58989285, 213.41963136],
                           [ 0, 0, 1 ]] )
        self.D = np.array([[-0.03146083],
                           [-0.03191633],
                           [ 0.05678013],
                           [-0.04003636]])
        
        
        


    def update_state_vector(self, state_vector, time):
        state_vector_dict = state_vector.interpolate(time)
        self.x_pos = state_vector_dict['x_pos']
        self.y_pos = state_vector_dict['y_pos']
        self.z_pos = state_vector_dict['z_pos']

        self.theta = state_vector_dict['theta']
        self.psi = state_vector_dict['psi']
        self.phi = state_vector_dict['phi']
        # if state_vector_dict['psi'] < 0:
        #     self.psi = 2 * np.pi + state_vector_dict['psi']
        # else:
        #     self.psi = state_vector_dict['psi']
        # if state_vector_dict['phi'] < 0:
        #     self.psi = 2 * np.pi + state_vector_dict['phi']
        # else:
        #     self.phi = state_vector_dict['phi']
        return None

    def update_camera_rotation_matrix(self):
            # Converting from [-pi, pi] to [0, 2pi]:
            if self.phi < 0:
                roll = 2 * np.pi + self.phi
            else:
                roll = self.phi
            if self.psi < 0:
                yaw = 2 * np.pi + self.psi
            else:
                yaw = self.psi
            pitch = self.theta
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
        
    def update_rotation_vector(self):
        R = self.update_camera_rotation_matrix()
        rvec, _ = cv2.Rodrigues(R)
        return rvec

    def update_camera_translation_vector(self):
        T = np.array([[self.x_pos], [self.y_pos], [self.z_pos]])
        return T

    def point3DWorld_to_point3D_Drone(self, point_3D):
        
        T = self.update_camera_translation_vector()
        R = self.update_camera_rotation_matrix()

        point_3D = point_3D[0][0]
        T = T.T[0]


        point_3D_translated = point_3D - T
        point_3D_translated = point_3D_translated[np.newaxis, :]
        point_3D_translated = point_3D_translated.T


        point_3D_rotated = R @ point_3D_translated

        
        return point_3D_rotated

    def project_3D_to_2D(self, points_3D_World_XYZ_RGB_Array, fisheye_bool = False):
        # (N, 6) array with columns [X, Y, Z, R, G, B].
        # Extract X, Y, Z
        rvec_null = np.zeros((3, 1), dtype=np.float32)
        Tvec_null = np.zeros((3, 1), dtype=np.float32)

        points_2D_XYRGB_array = np.zeros((points_3D_World_XYZ_RGB_Array.shape[0], 5))
        points_3D_drone_XYZRGB_array = np.zeros((points_3D_World_XYZ_RGB_Array.shape[0], 6))

        for i in range((points_3D_World_XYZ_RGB_Array.shape[0])):
            # Extract XYZ
            #print('points_3D_World_XYZ_RGB_Array', points_3D_World_XYZ_RGB_Array[i])
            points_3D_World = np.array([points_3D_World_XYZ_RGB_Array[i][0][:3]])
            #print('points_3D_World', points_3D_World)
            points_3D_World = np.array(points_3D_World, dtype=np.float32).reshape(-1, 1, 3)
            #print('points_3D_World reshaped', points_3D_World)

            # Apply transformation to project points from world to camera coordinates
            # The translation needs to be negated because we are moving the points to the camera's coordinate system
            points_3D_drone = self.point3DWorld_to_point3D_Drone(points_3D_World)
            #print('points_3D_camera', points_3D_camera) 
            points_3D_drone = np.array(points_3D_drone, dtype=np.float32).reshape(-1, 1, 3)
            #print('points_3D_camera reshaped', points_3D_camera)
            # Pause run untill

            #D_truncated = self.D[:, :4]
            if fisheye_bool:
                points_2D, _ = cv2.fisheye.projectPoints(points_3D_camera, rvec_null, Tvec_null, self.K, self.D)
            else:
                points_2D, _ = cv2.projectPoints(points_3D_camera, rvec_null, Tvec_null, self.K_nonfisheye, self.D_nonfisheye)
                # This gives awfull values.
            
            points_2D = points_2D.reshape(-1, 2)

            # Reattach the RGB values to the projected 2D points
            RBG_values = np.array(points_3D_World_XYZ_RGB_Array[i][0][3:])
            points_2D_RGB = np.concatenate((points_2D[0], RBG_values))
            #print('points_2D_RGB', points_2D_RGB)
            # Reattach the RGB values to the projected 3D points
            points_3D_drone_RGB = np.concatenate((points_3D_drone[0][0], RBG_values))
            #print('points_3D_camera_RGB', points_3D_camera_RGB)
            
            # Append the projected 2D points to the array
            points_2D_XYRGB_array[i] = points_2D_RGB
            points_3D_drone_XYZRGB_array[i] = points_3D_drone_RGB


        return points_2D_XYRGB_array, points_3D_drone_XYZRGB_array


'''
Non Fisheye
points_3D_World_XYZ_RGB_Array [[3.5000e+00 3.5000e+00 4.6875e-02 0.0000e+00 2.5500e+02 0.0000e+00]]
points_3D_World [[3.5      3.5      0.046875]]
points_3D_World reshaped [[[3.5      3.5      0.046875]]]
points_3D_camera [[0.89479161]
 [5.62572748]
 [1.10271088]]
points_3D_camera reshaped [[[0.8947916 5.6257277 1.1027108]]]
points_2D_RGB [7.3993492e+09 6.7103715e+10 0.0000000e+00 2.5500000e+02 0.0000000e+00]
points_3D_camera_RGB [  0.8947916   5.6257277   1.1027108   0.        255.          0.       ]



points_3D_World_XYZ_RGB_Array [[3.5000e+00 3.5000e+00 4.6875e-02 0.0000e+00 2.5500e+02 0.0000e+00]]
points_3D_World [[3.5      3.5      0.046875]]
points_3D_World reshaped [[[3.5      3.5      0.046875]]]
points_3D_camera [[0.89479161]
 [5.62572748]
 [1.10271088]]
points_3D_camera reshaped [[[0.8947916 5.6257277 1.1027108]]]
points_2D_RGB [314.10422 518.844     0.      255.        0.     ]
points_3D_camera_RGB [  0.8947916   5.6257277   1.1027108   0.        255.          0.       ]
'''
    

        

class StateVector:
    def __init__(self, file_path):
        self.time_array = []
        self.x_pos_array = []
        self.y_pos_array = []
        self.z_pos_array = []
        self.theta_array = []
        self.psi_array = []
        self.phi_array = []
        self.update_state_arrays(file_path)

        self.x_pos = self.x_pos_array[0]
        self.y_pos = self.y_pos_array[0]
        self.z_pos = self.z_pos_array[0]
        self.theta = self.theta_array[0]
        self.psi = self.psi_array[0]
        self.phi = self.phi_array[0]

    def update_state_arrays(self, file_path):
        df = pd.read_csv(file_path)

        # Convert the DataFrame into a dictionary of arrays
        data = df.to_dict('list')

        # Now data is a dictionary where the keys are the column names and the values are lists of column values
        self.time_array = data['time']
        self.x_pos_array = data['pos_x']
        self.y_pos_array = data['pos_y']
        self.z_pos_array = data['pos_z']
        self.theta_array = data['att_theta']
        self.psi_array = data['att_psi']
        self.phi_array = data['att_phi']

        return None

    def update_state_with_time(self, time):
        index = self.time_array.index(time)
        self.x_pos = self.x_pos_array[index]
        self.y_pos = self.y_pos_array[index]
        self.z_pos = self.z_pos_array[index]
        self.theta = self.theta_array[index]
        self.psi = self.psi_array[index]
        self.phi = self.phi_array[index]
        return None

    def update_state_with_index(self, index):
        self.x_pos = self.x_pos_array[index]
        self.y_pos = self.y_pos_array[index]
        self.z_pos = self.z_pos_array[index]
        self.theta = self.theta_array[index]
        self.psi = self.psi_array[index]
        self.phi = self.phi_array[index]
        return None
    
    def plot_xyz(self):
        fig, axs = plt.subplots(3)
        fig.suptitle('Position')
        axs[0].plot(self.time_array, self.x_pos_array)
        axs[0].set_title('X Position')
        axs[1].plot(self.time_array, self.y_pos_array)
        axs[1].set_title('Y Position')
        axs[2].plot(self.time_array, self.z_pos_array)
        axs[2].set_title('Z Position')
        plt.show()

    def plot_xyz_3d(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(self.x_pos_array, self.y_pos_array, self.z_pos_array)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()
    
    def plot_angles(self):
        fig, axs = plt.subplots(3)
        fig.suptitle('Angles')
        axs[0].plot(self.time_array, self.theta_array)
        axs[0].set_title('Theta')
        axs[1].plot(self.time_array, self.psi_array)
        axs[1].set_title('Psi')
        axs[2].plot(self.time_array, self.phi_array)
        axs[2].set_title('Phi')
        plt.show()

    def number_of_rows(self):
        return len(self.time_array)
    
    def return_points3d(self, index):
        return [self.x_pos_array[index], self.y_pos_array[index], self.z_pos_array[index]]
    
    def find_frequency(self):
        time_diff = [self.time_array[i] - self.time_array[i-1] for i in range(1, len(self.time_array))]
        return 1/np.mean(time_diff)
    
    def interpolate(self, time):
        # Interpolate the arrays with time array and output corresponding value
        interpolated_state_vector = {}
        interpolated_state_vector['time'] = time
        interpolated_state_vector['x_pos'] = np.interp(time, self.time_array, self.x_pos_array)
        interpolated_state_vector['y_pos'] = np.interp(time, self.time_array, self.y_pos_array)
        interpolated_state_vector['z_pos'] = np.interp(time, self.time_array, self.z_pos_array)
        interpolated_state_vector['theta'] = np.interp(time, self.time_array, self.theta_array)
        interpolated_state_vector['psi'] = np.interp(time, self.time_array, self.psi_array)
        interpolated_state_vector['phi'] = np.interp(time, self.time_array, self.phi_array)
        return interpolated_state_vector

        


class CyberZooStructure:
    def __init__(self, zmin):
        self.cyberzoo_green_width = 7
        self.cyberzoo_green_length = 7
        self.z_min = zmin
        self.corner_coordinates = {
            'A': {'x' : self.cyberzoo_green_width/2,
                'y' : self.cyberzoo_green_length/2,
                'z' : self.z_min},
            'B': {'x' : -self.cyberzoo_green_width/2,
                'y' : self.cyberzoo_green_length/2,
                'z' : self.z_min},
            'C': {'x' : -self.cyberzoo_green_width/2,
                'y' : -self.cyberzoo_green_length/2,
                'z' : self.z_min},
            'D': {'x' : self.cyberzoo_green_width/2,
                'y' : -self.cyberzoo_green_length/2,
                'z' : self.z_min}
        }
        self.points3d_A = [self.corner_coordinates['A']['x'], self.corner_coordinates['A']['y'], self.corner_coordinates['A']['z']]
        self.points3d_B = [self.corner_coordinates['B']['x'], self.corner_coordinates['B']['y'], self.corner_coordinates['B']['z']]
        self.points3d_C = [self.corner_coordinates['C']['x'], self.corner_coordinates['C']['y'], self.corner_coordinates['C']['z']]
        self.points3d_D = [self.corner_coordinates['D']['x'], self.corner_coordinates['D']['y'], self.corner_coordinates['D']['z']]

        # Define the RGB color codes for each corner
        self.corner_colors = {
            'A': (0, 255, 0),  # Green
            'B': (255, 0, 0),  # Red
            'C': (0, 0, 255),  # Blue
            'D': (255, 255, 0)  # Yellow
        }

        self.wall_height = 4 #m
        self.corner_upper_coordinates = {
            'A': {'x' : self.cyberzoo_green_width/2,
                'y' : self.cyberzoo_green_length/2,
                'z' : self.z_min + self.wall_height},
            'B': {'x' : -self.cyberzoo_green_width/2,
                'y' : self.cyberzoo_green_length/2,
                'z' : self.z_min + self.wall_height},
            'C': {'x' : -self.cyberzoo_green_width/2,
                'y' : -self.cyberzoo_green_length/2,
                'z' : self.z_min + self.wall_height},
            'D': {'x' : self.cyberzoo_green_width/2,
                'y' : -self.cyberzoo_green_length/2,
                'z' : self.z_min + self.wall_height}
        }
    
    def plot_cyberzoo(self):
        #FLOOR
        # Extract the corner coordinates
        corners = self.corner_coordinates
        # Create arrays for x, y, and z coordinates
        x = np.array([corners[corner]['x'] for corner in corners])
        y = np.array([corners[corner]['y'] for corner in corners])
        z = np.array([corners[corner]['z'] for corner in corners])
        # Create a meshgrid for the surface
        X, Y = np.meshgrid(x, y)
        Z = np.full_like(X, self.z_min)

        # Left wall : A, D, A_upper, D_upper
        corners_upper = self.corner_upper_coordinates
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

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()

    def return_points3d(self):
        return [self.points3d_A, self.points3d_B, self.points3d_C, self.points3d_D]
    
    def generate_line_points(self, start, end, num_points=100):
        x_values = [start['x'] + (end['x'] - start['x']) * i / (num_points - 1) for i in range(num_points)]
        y_values = [start['y'] + (end['y'] - start['y']) * i / (num_points - 1) for i in range(num_points)]
        z_values = [start['z'] + (end['z'] - start['z']) * i / (num_points - 1) for i in range(num_points)]
        return [[x, y, z] for x, y, z in zip(x_values, y_values, z_values)]

    def get_perimeter_points(self):
        # Lines between A -> B -> C -> D -> A
        AB_points = self.generate_line_points(self.corner_coordinates['A'], self.corner_coordinates['B'])
        BC_points = self.generate_line_points(self.corner_coordinates['B'], self.corner_coordinates['C'])
        CD_points = self.generate_line_points(self.corner_coordinates['C'], self.corner_coordinates['D'])
        DA_points = self.generate_line_points(self.corner_coordinates['D'], self.corner_coordinates['A'])

        # Combine all points, avoiding duplication of corner points
        perimeter_points = AB_points + BC_points[1:] + CD_points[1:] + DA_points[1:]
        return perimeter_points
    
    # Define a function to blend colors based on the proportion of the distance between two corners
    def blend_colors(self, color1, color2, blend_factor):
        return tuple(color1[i] + (color2[i] - color1[i]) * blend_factor for i in range(3))

    # Add color information to each point based on its position along each line
    def add_color_to_points(self, points, start_color, end_color):
        num_points = len(points)
        colored_points = []
        for i, point in enumerate(points):
            blend_factor = i / (num_points - 1)
            color = self.blend_colors(start_color, end_color, blend_factor)
            colored_points.append((point, color))
        return colored_points

    def get_colored_perimeter_points(self):
        # Get colored points for each line
        AB_colored_points = self.add_color_to_points(self.generate_line_points(self.corner_coordinates['A'], self.corner_coordinates['B']), self.corner_colors['A'], self.corner_colors['B'])
        BC_colored_points = self.add_color_to_points(self.generate_line_points(self.corner_coordinates['B'], self.corner_coordinates['C'])[1:], self.corner_colors['B'], self.corner_colors['C'])
        CD_colored_points = self.add_color_to_points(self.generate_line_points(self.corner_coordinates['C'], self.corner_coordinates['D'])[1:], self.corner_colors['C'], self.corner_colors['D'])
        DA_colored_points = self.add_color_to_points(self.generate_line_points(self.corner_coordinates['D'], self.corner_coordinates['A'])[1:], self.corner_colors['D'], self.corner_colors['A'])

        # Combine all colored points
        colored_perimeter_points = AB_colored_points + BC_colored_points + CD_colored_points + DA_colored_points

        # Convert to a more friendly format for display
        formatted_colored_points = np.array([(*point, *color) for point, color in colored_perimeter_points]) # (X, Y, Z, R, G, B)

        return formatted_colored_points


class VideoFeed:
    def __init__(self, frames_dir):
        self.frames_dir = frames_dir
        self.frame_files = sorted(os.listdir(frames_dir))
        self.images = glob.glob(frames_dir + '/*.jpg')
        self.index = 0
        self.image_current = self.image_read(self.index)

    def image_read(self, index = 0):
        self.index = index
        img = cv2.imread(os.path.join(self.frames_dir, self.frame_files[self.index]))
        self.image_current = img
        return img
    
    def image_rotate_90_counter(self):
        self.image_current = cv2.rotate(self.image_current, cv2.ROTATE_90_COUNTERCLOCKWISE)  # Rotate the image data
        return None

    
    def image_show(self, waitKeyvalue = 100, max_size  = 1500):
        cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
        
        # Get the original image size
        height, width = self.image_current.shape[:2]
        
        # Calculate the new size while keeping the same aspect ratio
        scale = max_size / max(height, width)
        new_width = int(scale * width)
        new_height = int(scale * height)
        
        cv2.resizeWindow('Image', new_width, new_height)
        cv2.imshow('Image', self.image_current)
        cv2.waitKey(waitKeyvalue)
        return None
    '''
    def draw_circle(self, x_coordinate, y_coordinate, radius=100, color=(0, 255, 0), thickness=-1):
        cv2.circle(self.image_current, (int(x_coordinate), int(y_coordinate)), radius, color, thickness)
        return None
    '''
    def draw_circle(self, points_2D_RGB, radius=100, thickness=-1):
        for point in points_2D_RGB:
            x_coordinate, y_coordinate = point[:2]
            color = tuple([int(c) for c in point[2:5]])
            #print('color:', color, 'type:', type(color))
            cv2.circle(self.image_current, (int(x_coordinate), int(y_coordinate)), radius, color, thickness)
        return None


    
    def number_of_images(self):
        return len(self.images)


    def filter(self, kernel_size = 3, sigma_Gaussian = 5, filter_type = 'Gaussian', 
                  d  = 9, sigmaColor = 75, sigmaSpace = 75):
        '''
        The aim of this function is to convert the frames to greyscale images and apply a Gaussian filter to them.

        https://pyimagesearch.com/2021/04/28/opencv-smoothing-and-blurring/

        inputs:
        frame_raw: the raw image
        kernel_size: the size of the kernel for the Gaussian filter, keep it odd: 3, 5, 7
        sigma_Gaussian: the standard deviation of the Gaussian filter

        outputs:
        frame_greyscale_filter: the filtered greyscale image
        '''
        # Convert the images to grayscale
        self.image_current = cv2.cvtColor(self.image_current, cv2.COLOR_BGR2GRAY)
        if filter_type == 'Gaussian':
            self.image_current = cv2.GaussianBlur(self.image_current, dst=None, ksize=(kernel_size, kernel_size), sigmaX= sigma_Gaussian)
        elif filter_type == 'Average':
            self.image_current = cv2.blur(self.image_current, (kernel_size, kernel_size))
        elif filter_type == 'Median':
            self.image_current = cv2.medianBlur(self.image_current, kernel_size)
        elif filter_type == 'Bilateral':
            self.image_current = cv2.bilateralFilter(self.image_current, d, sigmaColor, sigmaSpace)
        else:
            self.image_current = cv2.GaussianBlur(self.image_current, dst=None, ksize=(kernel_size, kernel_size), sigmaX= sigma_Gaussian)
        return None
    
    def resize_frame(self, scale_percent = 50):
        frame = self.image_current
        # Calculate the 50 percent of original dimensions
        width = int(frame.shape[1] * scale_percent / 100)

        # Calculate the 50 percent of original dimensions
        height = int(frame.shape[0] * scale_percent / 100)

        # dsize
        dsize = (width, height)

        # resize image
        frame_resized = cv2.resize(frame, dsize)
        self.image = frame_resized
        return None
    
    def find_time(self):
        # XXXXXXXX.jpg -> SS.XXXXXX jpg SS is seconds i.e. 60483805.jpg -> 60.483805 seconds
        time = int(self.frame_files[self.index].split('.')[0])/1000000
        return time
    



class OpticalFlow:
    def __init__(self, frame1, frame2):
        self.frame1 = frame1
        self.frame2 = frame2
        self.flow = self.compute_flow(frame1, frame2)
        self.flow_rgb = self.get_flow_viz()

    def compute_flow(frame1, frame2, pyr_scale=0.75, levels=3, winsize=5,
                    iterations=3, poly_n=10, poly_sigma=1.2, flags=0):
        
        flow = cv2.calcOpticalFlowFarneback(frame1, frame2, None,
                                            pyr_scale=pyr_scale,
                                            levels=levels,
                                            winsize=winsize,
                                            iterations=iterations,
                                            poly_n=poly_n,
                                            poly_sigma=poly_sigma,
                                            flags=flags)
        return flow

    def get_flow_viz(self):
        """ Obtains BGR image to Visualize the Optical Flow 
            """
        flow = self.flow
        hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
        hsv[..., 1] = 255

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang*180/np.pi/2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        self.flow_rgb = rgb
        return None
    
    def motion_mask(self, motion_thresh=1, kernel=np.ones((7,7))):
        flow = self.flow
        flow_mag, flow_ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        motion_thresh = np.linspace(0.1, 1, flow_mag.shape[0]).reshape(-1, 1)
        
        motion_mask = np.uint8(flow_mag > motion_thresh) * 255
        motion_mask = cv2.erode(motion_mask, kernel, iterations=1)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    
        return motion_mask
    
    def mask_rgb(self, mask):
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    def green_filter(self, mask_rgb, lower_green=np.array([30, 30, 30]), upper_green=np.array([90, 255, 255]), kernel_size=(15,15), canny_thresholds=(100, 200)):
        # Convert BGR to HSV
        frame2_bgr = self.frame2
        hsv = cv2.cvtColor(frame2_bgr, cv2.COLOR_BGR2HSV)

        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        for i in range(hsv.shape[-1]):
            hsv[:,:,i] = clahe.apply(hsv[:,:,i])

        # Threshold the HSV image to get only green colors
        green_mask = cv2.inRange(hsv, lower_green, upper_green)

        # Define a kernel for the morphological operations
        kernel = np.ones(kernel_size, np.uint8)

        # Apply dilation and erosion to smooth the mask
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)

        # Invert the mask
        green_mask_inv = cv2.bitwise_not(green_mask)

        # Bitwise-AND mask and original image (optional, to highlight green areas)
        result = cv2.bitwise_and(frame2_bgr, frame2_bgr, mask=green_mask_inv)

        # Display the original and result images
        stack_frames = np.vstack((frame2_bgr, result))
        cv2.imshow('Green Removed', stack_frames)

        mask_rgb_no_green = cv2.bitwise_and(mask_rgb, mask_rgb, mask=green_mask_inv)

        # Apply edge detection to the green mask
        edges = cv2.Canny(green_mask, *canny_thresholds)

        # Filter out the wall lines
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # Display the edges
        cv2.imshow('Edges', edges)

        return edges, mask_rgb_no_green
        

