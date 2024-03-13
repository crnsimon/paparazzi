def update_camera_orientation(num, x_pos_camera, y_pos_camera, z_pos_camera, theta_camera, phi_camera, psi_camera, lines):
    # Update the data for the camera's position
    lines[0].set_data(np.array([x_pos_camera[num]]), np.array([y_pos_camera[num]]))
    lines[0].set_3d_properties(np.array([z_pos_camera[num]]))

    # Calculate the camera's orientation vectors
    R = calculate_rotation_matrix(theta_camera[num], phi_camera[num], psi_camera[num])
    
    # Orientation vector length
    vector_length = 0.5

    # Orientation vectors
    x_vector = R @ np.array([vector_length, 0, 0])
    y_vector = R @ np.array([0, vector_length, 0])
    z_vector = R @ np.array([0, 0, vector_length])

    # Update the orientation lines
    for i, vec in enumerate([x_vector, y_vector, z_vector]):
        lines[i + 1].set_data(np.array([x_pos_camera[num], x_pos_camera[num] + vec[0]]),
                             np.array([y_pos_camera[num], y_pos_camera[num] + vec[1]]))
        lines[i + 1].set_3d_properties(np.array([z_pos_camera[num], z_pos_camera[num] + vec[2]]))

    return lines


def calculate_rotation_matrix(theta, phi, psi):
    # Assuming theta is pitch, phi is roll, and psi is yaw
    # Conversion from [-pi, pi] to [0, 2pi]
    if phi < 0:
        phi = 2 * np.pi + phi
    if psi < 0:
        psi = 2 * np.pi + psi
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(phi), -np.sin(phi)],
                   [0, np.sin(phi), np.cos(phi)]])
    Ry = np.array([[np.cos(theta), 0, np.sin(theta)],
                   [0, 1, 0],
                   [-np.sin(theta), 0, np.cos(theta)]])
    Rz = np.array([[np.cos(psi), -np.sin(psi), 0],
                   [np.sin(psi), np.cos(psi), 0],
                   [0, 0, 1]])
    R = Rz @ Ry @ Rx
    return R


# Create a figure and a 3D axis
if animation_bool:
    # Create a new figure
    fig = plt.figure()

    # Create a 3D axis
    ax = fig.add_subplot(111, projection='3d')

    # Initial position and orientation lines for the camera
    line0 = ax.plot([x_pos_camera[0]], [y_pos_camera[0]], [z_pos_camera[0]], 'bo')[0]
    lines = [line0]

    # Orientation lines (X - red, Y - green, Z - blue)
    colors = ['r', 'g', 'b']
    for color in colors:
        lines.append(ax.plot([0, 0], [0, 0], [0, 0], color)[0])

    # Plot the cyberzoo corners
    ax.plot(points3d_cyberzoo[0][0], points3d_cyberzoo[0][1], points3d_cyberzoo[0][2], 'go')
    ax.plot(points3d_cyberzoo[1][0], points3d_cyberzoo[1][1], points3d_cyberzoo[1][2], 'ro')
    ax.plot(points3d_cyberzoo[2][0], points3d_cyberzoo[2][1], points3d_cyberzoo[2][2], 'bo')
    ax.plot(points3d_cyberzoo[3][0], points3d_cyberzoo[3][1], points3d_cyberzoo[3][2], 'yo')
    ax.invert_yaxis()
    ax.invert_zaxis()

    # Setting the axes properties
    # ax.set_xlim3d([min(x_pos_camera), max(x_pos_camera)])
    # ax.set_ylim3d([min(y_pos_camera), max(y_pos_camera)])
    # ax.set_zlim3d([min(z_pos_camera), max(z_pos_camera)])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Define the update function for the animation
    # Create the animation
    # Sort all values w.r.t time
    x_pos_array = np.array(x_pos_camera)
    y_pos_array = np.array(y_pos_camera)
    z_pos_array = np.array(z_pos_camera)
    theta_pos_array = np.array(theta_camera)
    phi_pos_array = np.array(phi_camera)
    psi_pos_array = np.array(psi_camera)

    x_pos_sorted = x_pos_array[sorted_indices]
    y_pos_sorted = y_pos_array[sorted_indices]
    z_pos_sorted = z_pos_array[sorted_indices]
    theta_pos_sorted = theta_pos_array[sorted_indices]
    phi_pos_sorted = phi_pos_array[sorted_indices]
    psi_pos_sorted = psi_pos_array[sorted_indices]

    # Highlight fourth wall
    ax.xaxis.set_pane_color((0, 1.0, 0, 1.0))

    # ani = FuncAnimation(fig, update_camera_orientation, len(x_pos_camera),
    #                     fargs=(x_pos_camera, y_pos_camera, z_pos_camera, theta_camera, phi_camera, psi_camera, lines),
    #                     interval=100, blit=False)

    ani = FuncAnimation(fig, update_camera_orientation, len(x_pos_camera),
                        fargs=(x_pos_sorted, y_pos_sorted, z_pos_sorted, theta_pos_sorted, phi_pos_sorted, psi_pos_sorted, lines),
                        interval=100, blit=False)

    # Save the animation
    writer = FFMpegWriter(fps=15, metadata=dict(artist='Jonny'), bitrate=1800)
    ani.save("CameraCoordinateSystem.mp4", writer=writer)

    # Display the plot
    plt.show()


# Plot the points3d_cyberzoo_0_camera_array eg points3d_cyberzoo_0_camera_array
# Convert to numpy array
points3d_cyberzoo_0_camera_array = np.array(points3d_cyberzoo_0_camera_array)
points3d_cyberzoo_1_camera_array = np.array(points3d_cyberzoo_1_camera_array)
points3d_cyberzoo_2_camera_array = np.array(points3d_cyberzoo_2_camera_array)
points3d_cyberzoo_3_camera_array = np.array(points3d_cyberzoo_3_camera_array)
# Animation of the points3d_cyberzoo_0_camera_array
if animation_bool:

# Initialize the figure and axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Initialize the plots
line0, = ax.plot([], [], [], 'ro')
line1, = ax.plot([], [], [], 'go')
line2, = ax.plot([], [], [], 'mo')
line3, = ax.plot([], [], [], 'co')

# Set the limits of the plot
ax.set_xlim([min(points3d_cyberzoo_0_camera_array[:,0]), max(points3d_cyberzoo_0_camera_array[:,0])])
ax.set_ylim([min(points3d_cyberzoo_0_camera_array[:,1]), max(points3d_cyberzoo_0_camera_array[:,1])])
ax.set_zlim([min(points3d_cyberzoo_0_camera_array[:,2]), max(points3d_cyberzoo_0_camera_array[:,2])])

# Update function for the animation
def update(num):
    line0.set_data(points3d_cyberzoo_0_camera_array[:num, 0], points3d_cyberzoo_0_camera_array[:num, 1])
    line0.set_3d_properties(points3d_cyberzoo_0_camera_array[:num, 2])
    line1.set_data(points3d_cyberzoo_1_camera_array[:num, 0], points3d_cyberzoo_1_camera_array[:num, 1])
    line1.set_3d_properties(points3d_cyberzoo_1_camera_array[:num, 2])
    line2.set_data(points3d_cyberzoo_2_camera_array[:num, 0], points3d_cyberzoo_2_camera_array[:num, 1])
    line2.set_3d_properties(points3d_cyberzoo_2_camera_array[:num, 2])
    line3.set_data(points3d_cyberzoo_3_camera_array[:num, 0], points3d_cyberzoo_3_camera_array[:num, 1])
    line3.set_3d_properties(points3d_cyberzoo_3_camera_array[:num, 2])
    return line0, line1, line2, line3,

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=range(len(points3d_cyberzoo_0_camera_array)), blit=True)
# Save the animation
writer = FFMpegWriter(fps=15, metadata=dict(artist='MAVGroup'), bitrate=1800)
ani.save("COMPUTERVISIONSIM/Projection/CyberzooCornersInCameraCoordinateFrame.mp4", writer=writer)
# Show the animation
plt.show()