import numpy as np

yaw = -15 * np.pi / 180
Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                        [np.sin(yaw), np.cos(yaw), 0],
                        [0, 0, 1]])


NED = np.array([[[3.5,      3.5,      0.046875]]])

NED_North = NED[0][0][0]
NED_East = NED[0][0][1]
NED_Down = NED[0][0][2]


NED_T = [[NED_North],
         [NED_East],
         [NED_Down]]

World_XYZ_wrongnumpy = Rz @ NED_T

World_X = World_XYZ_wrongnumpy[0][0]
World_Y = World_XYZ_wrongnumpy[1][0]
World_Z = World_XYZ_wrongnumpy[2][0]

World = np.array([[[World_X, World_Y, World_Z]]])