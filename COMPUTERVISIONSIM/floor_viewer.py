import os
import cv2

#frames_dir = "data/cyberzoo_poles_panels"
frames_dir = "Data_gitignore/AE4317_2019_datasets/cyberzoo_poles_panels_mats_bottomcam"

for j in os.listdir(frames_dir):
    folder = os.path.join(frames_dir, j)

    folder2 = os.path.join(frames_dir2, j)

    if os.path.isdir(folder):
        # Get a list of image file names sorted in ascending order
        frame_files = sorted(os.listdir(folder))
        for i in range(0, len(frame_files)-1):
            #  STEP 1: read & rotate the frames
            frame1_bgr = cv2.imread(os.path.join(folder, frame_files[i]))
            frame1_bgr = cv2.rotate(frame1_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
            frame2_bgr = cv2.imread(os.path.join(folder, frame_files[i+1]))
            frame2_bgr = cv2.rotate(frame2_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)

            # Display frames
            cv2.imshow('frame1', frame1_bgr)
            cv2.imshow('frame2', frame2_bgr)
            cv2.waitKey(0)
