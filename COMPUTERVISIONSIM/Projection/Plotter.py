'''
Plotter file for projection
'''
import numpy as np
import matplotlib.pyplot as plt

def full_plotter(times_list, x_pos_camera, y_pos_camera, z_pos_camera, state_vector):
    plot_drone_in_cyberzoo(times_list, x_pos_camera, y_pos_camera, z_pos_camera)
    plot_state_vectors(state_vector)


# PLOT 1: Drone Frame
def plot_drone_in_cyberzoo(times_list, x_pos_camera, y_pos_camera, z_pos_camera):

    times = np.array(times_list)
    sorted_indices = np.argsort(times)
    times_sorted = times[sorted_indices]

    x_pos, y_pos, z_pos = np.array(x_pos_camera), np.array(y_pos_camera), np.array(z_pos_camera)
    x_pos_sorted = x_pos[sorted_indices]
    y_pos_sorted = y_pos[sorted_indices]
    z_pos_sorted = z_pos[sorted_indices]

    fig, axs = plt.subplots(3)
    fig.suptitle('Drone Coordinates in Cyberzoo Frame')
    axs[0].plot(times_sorted, x_pos_sorted)
    axs[0].set_title('X')
    axs[1].plot(times_sorted, y_pos_sorted)
    axs[1].set_title('Y')
    axs[2].plot(times_sorted, z_pos_sorted)
    axs[2].set_title('Z')
    plt.show()



# PLOT STATE VECTORS
def plot_state_vectors(state_vector):
    state_vector.plot_xyz_3d()
    state_vector.plot_angles()