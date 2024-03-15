import cv2
import numpy as np


K_nonfisheye = np.array([[1.06848861e+03, 0.00000000e+00, 2.43338041e+02],
                              [0.00000000e+00, 1.54122474e+03, 1.22209857e+02],
                              [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    
D_nonfisheye = np.array([[7.49344678e-01, -5.89979880e+01, 1.67460462e-01, -7.29040201e-02, 4.51276257e+02]])
        
K = np.array([[323.94986777, 0, 265.6212057 ],
            [ 0, 324.58989285, 213.41963136],
            [ 0, 0, 1 ]] )

D = np.array([[-0.03146083],
            [-0.03191633],
            [ 0.05678013],
            [-0.04003636]])



# Undistort the image
img = cv2.imread(image_files[0])
img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

img_undistorted = cv2.undistort(img, K, D, None, K)
