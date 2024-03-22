
import numpy as np
import cv2
import os
import glob

# Define the dimensions of the chessboard to be used
chessboard_size = (9, 6)  # Number of inner corners per chessboard row and column

# Arrays to store object points and image points from all images
objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image plane

# Prepare object points (0, 0, 0), (1, 0, 0), (2, 0, 0) ..., (6, 5, 0)
objp = np.zeros((np.prod(chessboard_size), 3), dtype=np.float32)
objp[:, :2] = np.indices(chessboard_size).T.reshape(-1, 2) #* 35.15 # checkboard size: 35.15mm

# Path to the folder containing chessboard images
folder_path = 'Data_gitignore/AE4317_2019_datasets/calibration_frontcam/20190121-163447'

# Load all image file paths
image_files = glob.glob(os.path.join(folder_path, '*.jpg'))

for image_file in image_files:
    img = cv2.imread(image_file)
    # Rotate image 90 degrees counter clockwise
    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
        cv2.imshow('Chessboard Corners', img)
        cv2.waitKey(1)

cv2.destroyAllWindows()

fisheyecamera = True
if not fisheyecamera:

    # Perform camera calibration
    ret, K, D, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # Display the camera matrix and distortion coefficients
    print("Camera matrix:\n", K)
    print("Distortion coefficients:\n", D)

else:
    # Calibrate fisheye camera
    N_OK = len(objpoints)
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]

    # Convert objpoints to a numpy array of type float32 and reshape to have 3 channels
    objpoints = [np.array(op, dtype=np.float32).reshape(-1, 1, 3) for op in objpoints]


    retval, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
        objpoints,
        imgpoints,
        gray.shape[::-1],
        K,
        D,
        rvecs,
        tvecs,
        cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    )

    print("Fisheye camera matrix:\n", K)
    print("Fisheye distortion coefficients:\n", D)

# Undistort the image
img = cv2.imread(image_files[0])
img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

img_undistorted = cv2.undistort(img, K, D, None, K)
# Show the original and undistorted images
cv2.imshow('Original', img)
cv2.imshow('Undistorted', img_undistorted)
cv2.waitKey(0)

# Re-projection error
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.fisheye.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, D)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    mean_error += error

print("Total error: ", mean_error / len(objpoints))