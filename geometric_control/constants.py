import numpy as np
import math
from math import sqrt


# Crazyflie mass and inertia

m = 0.080
J = (10**-3) * np.array([[6.679, 0.00, 0.00],
              [0.00, 6.679, 0.00],
              [0.00, 0.00, 13.13]])


g = 9.81
l = 0.0326695
d = sqrt(2) * l
c_tauf = 0.005964552

# mixing matrix

mat = np.array([[1, 1, 1, 1], [0, d, 0, -d], [-d, 0, d, 0], [-c_tauf, c_tauf, -c_tauf, c_tauf]])
mat_inv = np.linalg.inv(mat)
a0 = 4 * 1.563383 * (10**-5)

# control gains

kR = 8.81 # attitude gains
kOm = 2.54 # attitude gains
kx = 16.*m # position gains
kv = 5.6*m # position gains

Kx = np.diag([0.08, 0.08, 0.12])
Kv = np.diag([0.06, 0.06, 0.07])
KR = np.diag([0.00065, 0.00065, 0.0007])
KOm = np.diag([0.00004, 0.00004, 0.00005])

