
import numpy as np
import cv2
import os
import glob

# Define the dimensions of the chessboard to be used
chessboard_size = (8, 6)  # Number of inner corners per chessboard row and column

# Arrays to store object points and image points from all images
objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image plane

# Prepare object points (0, 0, 0), (1, 0, 0), (2, 0, 0) ..., (6, 5, 0)
objp = np.zeros((np.prod(chessboard_size), 3), dtype=np.float32)
objp[:, :2] = np.indices(chessboard_size).T.reshape(-1, 2)

# Path to the folder containing chessboard images
folder_path = 'Data_gitignore/AE4317_2019_datasets/calibration_frontcam/20190121-163447'

# Load all image file paths
image_files = glob.glob(os.path.join(folder_path, '*.jpg'))

for image_file in image_files:
    img = cv2.imread(image_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
        cv2.imshow('Chessboard Corners', img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

# Perform camera calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Display the camera matrix and distortion coefficients
print("Camera matrix:\n", mtx)
print("Distortion coefficients:\n", dist)

'''
Camera matrix:
 [[589.98363697   0.         117.18359156]
 [  0.         600.54137529 261.48275908]
 [  0.           0.           1.        ]]
Distortion coefficients:
 [[-0.32043809  0.27653614 -0.06730844 -0.04503392 -2.50539621]]
'''
