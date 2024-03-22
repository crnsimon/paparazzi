import cv2
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os
import csv
import pandas as pd
import torch.nn as nn
import torch
from transformers import get_cosine_schedule_with_warmup

# Load the data
data = pd.read_csv('points.csv')

# Remove the 'tensor' prefix and convert to float
data = data.applymap(lambda x: float(x.replace('tensor(', '').replace(')', '')))

# Split the data into 3D and 2D points
input_3d = data[['x_3d', 'y_3d', 'z_3d']].values
target_2d = data[['x_2d', 'y_2d']].values

# Convert to tensors
input_3d = torch.tensor(input_3d, dtype=torch.float32)
target_2d = torch.tensor(target_2d, dtype=torch.float32)

# Print tensor shape
print('input_3d.shape:', input_3d.shape)
print('target_2d.shape:', target_2d.shape)
#input_3d.shape: torch.Size([197, 3])
#target_2d.shape: torch.Size([197, 2])
# print warning if any nan or inf
if torch.isnan(input_3d).any() or torch.isnan(target_2d).any() or torch.isinf(input_3d).any() or torch.isinf(target_2d).any():
    print('Warning: nan or inf found in input_3d or target_2d')


def apply_camera_model(predicted_params, points_3d):
    """
    Apply the camera model to project 3D points to 2D using predicted camera parameters.
    
    :param predicted_params: Tensor containing the predicted camera parameters [f_x, f_y, c_x, c_y, k1, k2, k3, p_1, p_2]
    :param points_3d: Tensor containing 3D points in camera coordinates, shape [N, 3]
    :return: Tensor containing the projected 2D points, shape [N, 2]
    """
    predicted_params = predicted_params.squeeze()

    f_x, f_y, c_x, c_y, k1, k2, k3, p_1, p_2 = predicted_params

    f_x =  f_x
    f_y = f_y 
    c_x = c_x 
    c_y = c_y 
    #print(f_x, f_y, c_x, c_y, k1, k2, k3, p_1, p_2)

    # Unpack K matrix
    
    # Normalize 3D points (X_c/Z_c, Y_c/Z_c)
    X_c = points_3d[:, 0]
    Y_c = points_3d[:, 1]
    Z_c = points_3d[:, 2] + 1e-6
    x_normalized = X_c / Z_c
    y_normalized = Y_c / Z_c
    
    # Compute r^2 = x_normalized^2 + y_normalized^2
    r_squared = x_normalized**2 + y_normalized**2
    
    # Apply radial distortion correction
    radial_factor = 1 + k1 * r_squared + k2 * r_squared**2 + k3 * r_squared**3
    x_radial = x_normalized * radial_factor
    y_radial = y_normalized * radial_factor
    
    # Apply tangential distortion correction
    x_tangential = 2 * p_1 * x_normalized * y_normalized + p_2 * (r_squared + 2 * x_normalized**2)
    y_tangential = p_1 * (r_squared + 2 * y_normalized**2) + 2 * p_2 * x_normalized * y_normalized
    
    # Combine radial and tangential distortions
    x_distorted = x_radial + x_tangential
    y_distorted = y_radial + y_tangential
    
    # Convert distorted coordinates back to pixel coordinates
    u = f_x * x_distorted + c_x
    v = f_y * y_distorted + c_y
    
    # Stack the 2D points into a single tensor
    points_2d = torch.stack([u, v], dim=-1)

    img_width = 240
    img_height = 520

    # Normalise the points
    points_2d[:, 0] = points_2d[:, 0] / img_width
    points_2d[:, 1] = points_2d[:, 1] / img_height

    
    return points_2d

# Make neural network
class CameraModel(nn.Module):
    def __init__(self):
        super(CameraModel, self).__init__()
        # Input is 3D points, output is predicted params size 9
        self.fc1 = nn.Linear(3, 9)
        self.fc2 = nn.Linear(9, 9)

        # initial guesses [500, 500, 300, 300, 0.01, 0.01, 0.01, 0.01, 0.01] of output change bias
        self.fc1.bias.data = torch.tensor([500, 500, 300, 300, 0.01, 0.01, 0.01, 0.01, 0.01], dtype=torch.float32)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
# Create the model
model = CameraModel()

# Define the loss function
criterion = nn.MSELoss()

# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.00000001)

# Define the learning rate scheduler
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=1000)

# Train the model
num_epochs = 1000
for epoch in range(num_epochs):
    # Forward pass
    
    predicted_parameters = model(input_3d)
    predicted_parameters_mean = torch.mean(predicted_parameters, dim=0).unsqueeze(0)
    predicted_2d = apply_camera_model(predicted_parameters_mean, input_3d)
    loss = criterion(predicted_2d, target_2d)

    # Backward pass
    optimizer.zero_grad()

    # Perform backpropagation
    loss.backward()

    # Update the weights
    optimizer.step()

    # Update the learning rate
    scheduler.step()

    # Print the loss every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')
