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
    """
    Generates a binary motion mask from the optical flow magnitude,
    highlighting areas of significant motion.

    Parameters:
    - flow_mag: An array representing the magnitude of optical flow between two frames,
        indicating motion intensity.
    - motion_thresh: A threshold value to determine significant motion.
        Pixels with flow magnitude above this value are considered in motion.
    - kernel: A matrix used for morphological operations, defining the neighborhood size
        and shape for these operations.

    Returns:
    - motion_mask: A binary image (mask) where pixels in motion are white (255)
        and static pixels are black (0).
    """
    # Create an initial binary mask by thresholding the flow magnitude.
    # Pixels with a magnitude greater than motion_thresh are set to 1, otherwise to 0.
    # This binary mask is then converted to an 8-bit format (0 or 255).
    motion_mask = np.uint8(flow_mag > motion_thresh) * 255

    # Apply erosion to reduce noise by removing small white regions in the mask.
    motion_mask = cv2.erode(motion_mask, kernel, iterations=1)

    # Apply opening (erosion followed by dilation) to remove small objects or noise
    #from the foreground.
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Apply closing (dilation followed by erosion) to fill in small holes
    # and gaps in white regions,
    # helping to create continuous areas of motion.
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    # Return the refined motion mask, where significant motion areas are highlighted.
    return motion_mask

def filter_wall_lines(edge_image):
    # Detect lines using Hough Line Transform
    lines = cv2.HoughLinesP(edge_image, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)
    
    # Create a mask with the same dimensions as the edge image, initialized to the max value (white)
    mask = np.ones(edge_image.shape, dtype=np.uint8) * 255
    
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                # Calculate the orientation of the line
                angle = np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi
                
                # Check if line is vertical or horizontal based on an angle threshold (e.g., +/- 10 degrees)
                if abs(angle) < 10 or abs(angle) > 170:
                    # Draw the line on the mask
                    cv2.line(mask, (x1, y1), (x2, y2), 0, thickness=2)
    
    # Remove the wall lines from the edge image by masking
    filtered_image = cv2.bitwise_and(edge_image, edge_image, mask=mask)
    
    return filtered_image

frames_dir = "Data_gitignore/AE4317_2019_datasets/cyberzoo_poles_panels"
#frames_dir = "Data_gitignore/AE4317_2019_datasets/cyberzoo_poles_panels_mats"
#frames_dir = "Data_gitignore/AE4317_2019_datasets/cyberzoo_poles_panels_mats_bottomcam"

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

            '''
            # Convert BGR to HSV
            hsv = cv2.cvtColor(frame2_bgr, cv2.COLOR_BGR2HSV)
            # Increase brightness
            value = 50
            lim = 255 - value
            hsv[:,:,2] = np.where((255 - hsv[:,:,2]) < value, 255, hsv[:,:,2]+value)
            # Define range of green color in HSV
            lower_green = np.array([30, 30, 30])  # Adjust these values according to your needs
            upper_green = np.array([90, 255, 255])
            # Threshold the HSV image to get only green colors
            mask = cv2.inRange(hsv, lower_green, upper_green)
            # Apply adaptive thresholding
            mask = cv2.adaptiveThreshold(mask, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
            # Bitwise-AND mask and original image (optional, to highlight green areas)
            result = cv2.bitwise_and(frame2_bgr, frame2_bgr, mask=mask)
            # Display the original and result images
            cv2.imshow('Original Image', frame2_bgr)
            cv2.imshow('Detected Green Areas', result)
            '''



            # STEP 2: resize & filter the frames
            frame1_resized = resize_frame(frame1_bgr, scale_percent = 100)
            frame1_filtered = filter_frames(frame1_resized)
            frame2_resized = resize_frame(frame1_bgr, scale_percent = 100)
            frame2_filtered = filter_frames(frame2_resized)

            # STEP 3: compute the dense optical flow            
            flow = compute_flow(frame1_filtered, frame2_filtered,
                                pyr_scale=0.75, levels=3, winsize=5,
                                iterations=3, poly_n=10, poly_sigma=1.2, flags=0)
            
            # STEP 4: optical flow visualization -  transforming the flow data into a human-readable form where motion directions and magnitudes are represented by colors.
            #rgb = get_flow_viz(flow)

            # STEP 5: separte the flow into magnitude and direction
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

            # Display mag
            #plt.imshow(mag, cmap='gray')

            # STEP 6: get the motion mask
            # get variable motion thresh based on prior knowledge of camera position
            motion_thresh = np.linspace(0.1, 1, mag.shape[0]).reshape(-1, 1)
            # get motion mask
            mask = get_motion_mask(mag, motion_thresh=motion_thresh)
            mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

            # STEP 7: draw contours
            gray = cv2.cvtColor(frame2_bgr, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour_image = frame2_bgr.copy()
            cv2.drawContours(mask_rgb, contours, -1, (0, 255, 0), 2)
            frame3 = cv2.Canny(frame2_bgr, 80, 150)

            # STEP 8: filter out the green
            # Convert BGR to HSV
            hsv = cv2.cvtColor(frame2_bgr, cv2.COLOR_BGR2HSV)
            '''
            # Apply histogram equalization
            for i in range(hsv.shape[-1]):
                hsv[:,:,i] = cv2.equalizeHist(hsv[:,:,i])
            '''
            # Or apply CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            for i in range(hsv.shape[-1]):
                hsv[:,:,i] = clahe.apply(hsv[:,:,i])

            # Define range of green color in HSV
            lower_green = np.array([30, 30, 30])  # Adjust these values according to your needs
            upper_green = np.array([90, 255, 255])

            # Threshold the HSV image to get only green colors
            green_mask = cv2.inRange(hsv, lower_green, upper_green)
            # Define a kernel for the morphological operations
            kernel = np.ones((15,15), np.uint8)  # Adjust the size of the kernel as needed

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


            # STEP 8 : display
            display = 1
            if display:
                stacked_frames = np.vstack((frame2_bgr, mask_rgb))
                #cv2.imshow('Frames with Optical Flow', stacked_frames)
                #cv2.imshow('Contours', frame3)
                # show mask_rgb above mask_rgb_no_green, in the same window
                stacked_masks = np.vstack((mask_rgb, mask_rgb_no_green))
                cv2.imshow('Masks', stacked_masks)

                # Exit if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break