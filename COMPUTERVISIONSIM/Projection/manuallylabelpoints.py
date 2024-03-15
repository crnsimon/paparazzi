import cv2
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os

plot_bool = False

# Callback function for mouse events
def click_event(event, x, y, flags, param):
    global clicked_points_left, clicked_points_right, img
    if event == cv2.EVENT_LBUTTONDOWN:  # Left mouse button clicked
        clicked_points_left.append((x, y))  # Append the (x, y) tuple to the list
        print(f'Coordinates: ({x}, {y})')  # Print coordinates of the click
        cv2.circle(img, (x, y), 5, (255, 0, 0), -1)  # Mark the clicked point with a blue circle
        cv2.imshow('image', img)  # Show the image with the marked point
    if event == cv2.EVENT_RBUTTONDOWN:  # Right mouse button clicked
        clicked_points_right.append((x, y))  # Append the (x, y) tuple to the list
        print(f'Coordinates: ({x}, {y})')  # Print coordinates of the click
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)  # Mark the clicked point with a red circle
        cv2.imshow('image', img)  # Show the image with the marked point
    return clicked_points_left, clicked_points_right

def clicker_call(image_path):
    global img, clicked_points_left, clicked_points_right
    # Load your image
    img = cv2.imread(image_path)
    # Rotate the image 90 counter
    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    cv2.imshow('image', img)

    # Set mouse callback function for 'image' window
    cv2.setMouseCallback('image', click_event)

    cv2.waitKey(0)  # Wait for a key press to exit
    cv2.destroyAllWindows()  # Close all OpenCV windows

    # Optionally, print all clicked points
    print("All clicked points left :", clicked_points_left)
    print("All clicked points right :", clicked_points_right)

    return clicked_points_left, clicked_points_right

# Linear fit over the clicked points


# Define the function to fit
def f(x, A, B):
    return A*x + B

def interpolator(clicked_points_side, plot_bool = False):
    # Fit the function to the data
    xdata = [p[0] for p in clicked_points_side]
    ydata = [p[1] for p in clicked_points_side]
    popt, pcov = curve_fit(f, xdata, ydata)
    # Print the fit parameters
    print("A =", popt[0], ", B =", popt[1])

    # Plot the data and the fit
    if plot_bool:
        plt.plot(xdata, ydata, 'bo', label='data')
        plt.plot(xdata, f(np.array(xdata), *popt), 'r-', label='fit: A=%5.3f, B=%5.3f' % tuple(popt))
        plt.legend()
        plt.show()
    return popt

def clicker_to_lines(image_path):
    clicked_points_left, clicked_points_right = clicker_call(image_path)
    popt_left = interpolator(clicked_points_left)
    popt_right = interpolator(clicked_points_right)
    return popt_left, popt_right

image_path = 'COMPUTERVISIONSIM/Projection/single_image/image/68016093.jpg'

images_path = r"C:\Users\Jonathan van Zyl\Documents\GitHub\paparazzi\Data_gitignore\AE4317_2019_datasets\cyberzoo_poles_panels\20190121-140205"
file_path = "Data_gitignore/AE4317_2019_datasets/cyberzoo_poles_panels/20190121-140303.csv"
frame_files = sorted(os.listdir(images_path))

# Now want to stores the popt_left and popt_right for each image
popt_left_all = []
popt_right_all = []

no_images = len(frame_files)
max_iters = 100
steps_between_image = int(no_images/max_iters)
stop_iter = steps_between_image*max_iters

print(0, stop_iter, steps_between_image)

for i in range(0, stop_iter, steps_between_image):
    clicked_points_left = []
    clicked_points_right = []
    image_path = os.path.join(images_path, frame_files[i])
    img = cv2.imread(image_path)
    popt_left, popt_right = clicker_to_lines(image_path)

    popt_left_all.append(popt_left)
    popt_right_all.append(popt_right)

    print(popt_left_all)
    print(popt_right_all)

    # Stop if q is pressed
    k = cv2.waitKey(0)
    if k == ord('q'):
        break



