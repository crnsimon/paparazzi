import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Create a window to display the frames
cv2.namedWindow('Frames with Optical Flow', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Frames with Optical Flow', 720, 720)

def filter_frames(frame_raw, kernel_size = 3, sigma_Gaussian = 5, filter_type = 'Gaussian', 
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
    frame_greyscale = cv2.cvtColor(frame_raw, cv2.COLOR_BGR2GRAY)
    if filter_type == 'Gaussian':
        frame_greyscale_filter = cv2.GaussianBlur(frame_greyscale, dst=None, ksize=(kernel_size, kernel_size), sigmaX= sigma_Gaussian)
    elif filter_type == 'Average':
        frame_greyscale_filter = cv2.blur(frame_greyscale, (kernel_size, kernel_size))
    elif filter_type == 'Median':
        frame_greyscale_filter = cv2.medianBlur(frame_greyscale, kernel_size)
    elif filter_type == 'Bilateral':
        frame_greyscale_filter = cv2.bilateralFilter(frame_greyscale, d, sigmaColor, sigmaSpace)
    else:
        frame_greyscale_filter = cv2.GaussianBlur(frame_greyscale, dst=None, ksize=(kernel_size, kernel_size), sigmaX= sigma_Gaussian)
    return frame_greyscale_filter

def resize_frame(frame, scale_percent = 50):
    '''
    USED TO REDUCE COMPUTATIONAL POWER
    The aim of this function is to resize the frames to a certain percentage.

    inputs:
    frame: the raw image
    scale_percent: the percentage of the original size

    outputs:
    frame_resized: the resized image
    '''
    # Calculate the 50 percent of original dimensions
    width = int(frame.shape[1] * scale_percent / 100)

    # Calculate the 50 percent of original dimensions
    height = int(frame.shape[0] * scale_percent / 100)

    # dsize
    dsize = (width, height)

    # resize image
    frame_resized = cv2.resize(frame, dsize)
    return frame_resized

def compute_flow(frame1, frame2, pyr_scale=0.75, levels=3, winsize=5,
                  iterations=3, poly_n=10, poly_sigma=1.2, flags=0):
    '''
    The aim of this function is to compute the optical flow between two frames using the Farneback method.

    Inputs:
    - frame1: The first 8-bit single-channel frame to calculate the flow from.
    - frame2: The second 8-bit single-channel frame to calculate the flow to.
    - pyr_scale: Image scale (<1) to build pyramids for each image
            pyr_scale=0.5 means a classical pyramid, where each next layer is twice smaller than the previous one.
    - levels: Number of pyramid layers including the initial image
            levels=1 means that no extra layers are created and only the original images are used.
    - winsize: Averaging window size; larger values increase the algorithm's robustness
            to image noise and give more chances for fast motion detection,
            but yield more blurred motion field.
    - iterations: Number of iterations the algorithm does at each pyramid level.
    - poly_n: Size of the pixel neighborhood used to find polynomial expansion in each pixel
            larger values mean that the image will be approximated with smoother surfaces,
            yielding more robust algorithm and more blurred motion field.
    - poly_sigma: Standard deviation of the Gaussian that is used to smooth derivatives
            used as a basis for the polynomial expansion; can be used to control the
            smoothness of the output.
    - flags: Operation flags that can be 0 or a combination of the following flags:
        OPTFLOW_USE_INITIAL_FLOW uses the input flow as an initial flow approximation,
        OPTFLOW_FARNEBACK_GAUSSIAN uses the Gaussian winsize x winsize filter instead
        of a box filter of the same size for optical flow estimation;
        usually, this option gives z more accurate flow than with a box filter,
        at the cost of lower speed; normally, winsize for a Gaussian window should be
        set to a larger value to achieve the same level of robustness.

    Outputs:
    - flow: Computed flow image that has the same size as the input image and type CV_32FC2.

    '''
    flow = cv2.calcOpticalFlowFarneback(frame1, frame2, None,
                                        pyr_scale=pyr_scale,
                                        levels=levels,
                                        winsize=winsize,
                                        iterations=iterations,
                                        poly_n=poly_n,
                                        poly_sigma=poly_sigma,
                                        flags=flags)
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

#frames_dir = "data/cyberzoo_poles_panels"
frames_dir = "Data_gitignore/AE4317_2019_datasets/cyberzoo_poles_panels"

for j in os.listdir(frames_dir):
    folder = os.path.join(frames_dir, j)

    if os.path.isdir(folder):
        # Get a list of image file names sorted in ascending order
        frame_files = sorted(os.listdir(folder))
        for i in range(0, len(frame_files)-1):
            #  STEP 1: read & rotate the frames
            frame1_bgr = cv2.imread(os.path.join(folder, frame_files[i]))
            frame1_bgr = cv2.rotate(frame1_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
            frame2_bgr = cv2.imread(os.path.join(folder, frame_files[i+1]))
            frame2_bgr = cv2.rotate(frame2_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)


            # STEP 2: resize & filter the frames
            frame1_resized = resize_frame(frame1_bgr, scale_percent = 100)
            frame1_filtered = filter_frames(frame1_resized)
            frame2_resized = resize_frame(frame1_bgr, scale_percent = 100)
            frame2_filtered = filter_frames(frame2_resized)

            # STEP 3: compute the dense optical flow            
            flow = compute_flow(frame1_filtered, frame2_filtered,
                                pyr_scale=0.75, levels=3, winsize=5,
                                iterations=3, poly_n=10, poly_sigma=1.2, flags=0)

            # Contours

            gray = cv2.cvtColor(frame2_bgr, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour_image = frame2_bgr.copy()

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