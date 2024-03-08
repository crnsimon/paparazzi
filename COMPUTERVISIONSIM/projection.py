import numpy as np

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

# Example usage:
camera = Camera()
# Set camera parameters based on drone's state
camera.set_position(np.array([10, 10, 10]))  # Example position
camera.set_target(np.array([0, 0, 0]))  # Example target
camera.set_viewport(0, 0, 1920, 1080)  # Example viewport dimensions

point_3d = np.array([5, 5, 0])  # Example 3D point

# Project the 3D point onto the 2D image plane
point_2d = project_point(camera, point_3d)
print(f"Projected 2D point: {point_2d}")
