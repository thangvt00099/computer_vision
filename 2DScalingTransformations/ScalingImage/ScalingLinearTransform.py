import cv2
import numpy as np

# Matrix 2D
P = np.array([2, 4])
Sx, Sy = 3, 0
S = np.array([[Sx, 0], [0, Sy]])
print(S)

P_dash = S.dot(P)
print(P_dash)
print("=======")

# Matrix 3D
P2 = np.array([0, 2, 4])
Sx, Sy, Sz = 2, 2, 2
S2 = np.array([[Sx, 0, 0], [0, Sy, 0], [0, 0, Sz]])
P2_dash = S2.dot(P2)
print(P2_dash)