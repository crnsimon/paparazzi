import numpy as np
import glob
import matplotlib.pyplot as plt
import os



# Load images .jpg from Data_gitignore/AE4317_2019_datasets/calibration_bottomcam/20190121-163846

home_dir = os.path.expanduser("~")
image_dir = f'{home_dir}/paparazzi/COMPUTERVISIONSIM/Data_gitignore/AE4317_2019_datasets/calibration_bottomcam/20190121-163846'
image_files = glob.glob(image_dir + '/*.jpg')


# Process the images
i = 0
for image_file in image_files:
    image = plt.imread(image_file)
    # Add your image processing code here

    # Display the first image
    print(i)
    if i == 0:
        print('shown')
        plt.imshow(image)
        plt.axis('off')
        plt.show()
    i += 1

# Display the processed images
plt.show()