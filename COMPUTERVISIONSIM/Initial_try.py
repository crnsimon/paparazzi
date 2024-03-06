import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Create a window to display the frames
cv2.namedWindow('Frames with Optical Flow', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Frames with Optical Flow', 720, 720)


def compute_flow(frame1, frame2):
    # convert to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # blurr image
    gray1 = cv2.GaussianBlur(gray1, dst=None, ksize=(3,3), sigmaX=5)
    gray2 = cv2.GaussianBlur(gray2, dst=None, ksize=(3,3), sigmaX=5)

    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None,
                                        pyr_scale=0.75,
                                        levels=3,
                                        winsize=5,
                                        iterations=3,
                                        poly_n=10,
                                        poly_sigma=1.2,
                                        flags=0)
    return flow


def get_flow_viz(flow):
    """ Obtains BGR image to Visualize the Optical Flow 
        """
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 1] = 255

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang*180/np.pi/2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    return rgb

def get_motion_mask(flow_mag, motion_thresh=1, kernel=np.ones((7,7))):
    """ Obtains Detection Mask from Optical Flow Magnitude
        Inputs:
            flow_mag (array) Optical Flow magnitude
            motion_thresh - thresold to determine motion
            kernel - kernal for Morphological Operations
        Outputs:
            motion_mask - Binray Motion Mask
        """
    motion_mask = np.uint8(flow_mag > motion_thresh)*255

    motion_mask = cv2.erode(motion_mask, kernel, iterations=1)
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    return motion_mask

frames_dir = "data/cyberzoo_poles_panels"


for j in os.listdir(frames_dir):
    folder = os.path.join(frames_dir, j)

    if os.path.isdir(folder):
        # Get a list of image file names sorted in ascending order
        frame_files = sorted(os.listdir(folder))
        for i in range(0, len(frame_files)-1):
            # read frames
            frame1_bgr = cv2.imread(os.path.join(folder, frame_files[i]))
            frame2_bgr = cv2.imread(os.path.join(folder, frame_files[i+1]))
            frame1_bgr = cv2.rotate(frame1_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
            frame2_bgr = cv2.rotate(frame2_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)

            # Contours

            gray = cv2.cvtColor(frame2_bgr, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour_image = frame2_bgr.copy()

            # compute dense optical flow
            flow = compute_flow(frame1_bgr, frame2_bgr)

            # separate into magntiude and angle
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

            # get optical flow visualization
            rgb = get_flow_viz(flow)

            # get variable motion thresh based on prior knowledge of camera position
            motion_thresh = np.linspace(0.1, 1, mag.shape[0]).reshape(-1, 1)
            
            # get motion mask
            mask = get_motion_mask(mag, motion_thresh=motion_thresh)
            mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

            cv2.drawContours(mask_rgb, contours, -1, (0, 255, 0), 2)

            frame3 = cv2.Canny(frame2_bgr, 80, 150)

            # display
            stacked_frames = np.vstack((frame2_bgr, mask_rgb))
            cv2.imshow('Frames with Optical Flow', stacked_frames)
            cv2.imshow('Contours', frame3)
            # Exit if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break