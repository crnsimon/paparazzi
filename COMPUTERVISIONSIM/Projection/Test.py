import projection_functions as proj

cam = proj.Camera()
file_path = "C:/Users/aname/Documents/GitHub/paparazzi/AE4317_2019_datasets/AE4317_2019_datasets/cyberzoo_poles_panels/20190121-140303.csv"

state = proj.StateVector(file_path)
cam.update_state_vector(state)
print(cam.get_state())