import math
import numpy as np

point_3D = np.array([[[3.5, 3.5, 0.046875]]])
T = np.array([[0.078125], [-1.066406], [-1.003906]])
R = np.array([[0.88428642, -0.46694365, 0.00107375],
              [0.46693224, 0.88424321, -0.00939281],
              [0.00343646, 0.0088073, 0.99995531]])


point_3D = point_3D[0][0]
T = T.T[0]


point_3D_translated = point_3D - T
point_3D_translated = point_3D_translated[np.newaxis, :]
point_3D_translated = point_3D_translated.T


point_3D_rotated = R @ point_3D_translated

print(point_3D_rotated)