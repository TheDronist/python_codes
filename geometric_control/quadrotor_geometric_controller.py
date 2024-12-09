import numpy as np
from math import pi
from numpy.linalg import inv
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from constants import *
from scipy.spatial.transform import Rotation


def model(z, t, f, M):

    x = np.array([z[0], z[1], z[2]])
    v = np.array([z[3], z[4], z[5]])
    R = np.reshape(z[6:15],(3,3))
    Om = np.array([z[15], z[16],z[17]])

    e3 = np.array([0, 0, 1])
    Om_hat = hat_map(Om)
    b3 = np.dot(R,e3)
    J_inv = inv(J)
    
    dxdt = v
    dvdt = - g * e3 + f * b3 / m
    dRdt = np.dot(R, Om_hat)
    dOmdt = np.dot(J_inv, M - np.cross(Om, J.dot(Om)))
    dzdt = np.concatenate((dxdt, dvdt, dRdt.flatten(), dOmdt))

    return dzdt

def deriv_vect(a, a_d, a_dd):
    a_norm = np.linalg.norm(a)
    b = a / a_norm
    b_dot = a_d / a_norm - a * np.inner(a, a_d) / a_norm**3
    b_ddot = a_dd / a_norm - a_d / (a_norm**3) * (2 * np.inner(a, a_d)) - a / (a_norm**3) *(np.inner(a_d, a_d) + np.inner(a, a_dd)) + 3 * a / (a_norm**5) * (np.inner(a, a_d)**2)

    return b, b_dot, b_ddot

def hat_map(x):
    output = np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])
    return output

def vee_map(x):
    output = np.array([x[2,1], x[0,2], x[1,0]])
    return output

def err_Om(R, R_d, Om, Om_d):
    prod1 = np.dot(np.transpose(R),R_d)
    output = Om - np.dot(prod1,Om_d)
    return output


def lee_geom_cont(z, x_des, x_d_des, x_2d_des, x_3d_des, x_4d_des, b1_dir, b1_dir_dot, t):

    x = np.array([z[0], z[1], z[2]])
    v = np.array([z[3], z[4], z[5]])
    R = np.reshape(z[6:15],(3,3))
    Om = np.array([z[15], z[16],z[17]])

    e3 = np.array([0, 0, 1])
    Om_hat = hat_map(Om)
    b3 = np.dot(R,e3)
    b3_dot = np.linalg.multi_dot([R, Om_hat, e3])   

    b1_dir_ddot = (pi**2) * np.array([-np.cos(pi*t), -np.sin(pi*t), 0]) 

    ex = x - x_des
    ev = v - x_d_des
    f_vect = - kx*ex - kv*ev + m*g*e3 + m * x_2d_des
    f =  np.inner(f_vect, b3)
    ev_dot = - g * e3 + f / m * b3 - x_2d_des
    f_vect_dot = - kx * ev - kv * ev_dot + m * x_3d_des
    f_dot = np.inner(f_vect_dot, b3) + np.inner(f_vect, b3_dot)
    ev_ddot = f_dot / m * b3 + f / m * b3_dot - x_3d_des
    f_vect_ddot = - kx * ev_dot - kv * ev_ddot + m * x_4d_des

    b3_des, b3_des_dot, b3_des_ddot = deriv_vect(f_vect, f_vect_dot, f_vect_ddot)

    A2 = - np.cross(b1_dir, b3_des)
    A2_dot = - np.cross((b1_dir_dot), b3_des) - np.cross((b1_dir), b3_des_dot)
    A2_ddot = - np.cross((b1_dir_ddot), b3_des) - 2 * np.cross((b1_dir_dot), b3_des_dot) - np.cross((b1_dir), b3_des_ddot)

    b2_des, b2_des_dot, b2_des_ddot = deriv_vect(A2, A2_dot, A2_ddot)

    b1_des = np.cross(b2_des, b3_des)
    b1_des_dot = np.cross(b2_des_dot, b3_des) + np.cross(b2_des, b3_des_dot)
    b1_des_ddot = np.cross(b2_des_ddot, b3_des) + 2 * np.cross(b2_des_dot, b3_des_dot) + np.cross(b2_des, b3_des_ddot)

    R_des = np.array([b1_des, b2_des, b3_des]).T
    R_des_d = np.array([b1_des_dot, b2_des_dot, b3_des_dot]).T
    R_des_ddot = np.array([b1_des_ddot, b2_des_ddot, b3_des_ddot]).T

    Om_des = vee_map(np.dot(np.transpose(R_des),R_des_d))
    Om_des_d = vee_map(np.dot(np.transpose(R_des),R_des_ddot) - np.dot(hat_map(Om_des), hat_map(Om_des)))

    e_R = 0.5 * vee_map(np.dot(np.transpose(R_des),R) - np.dot(np.transpose(R),R_des))
    e_Om = err_Om(R, R_des, Om, Om_des)
    
    M = -kR * e_R - kOm * e_Om + np.cross(Om, J.dot(Om)) - np.linalg.multi_dot([J, Om_hat, np.transpose(R), R_des, Om_des]) + np.linalg.multi_dot([J, np.transpose(R), R_des, Om_des_d])

    return f, M

rpy = np.array([0, 0, 0])

x0 = np.array([0, 0, 0])
v0 = np.array([0, 0, 0])
Om0 = np.array([0, 0, 0])
r = Rotation.from_euler('xyz', rpy, degrees=True)
rot = r.as_matrix()
R0 = np.reshape(rot, (3, 3)).T
z0 = np.concatenate((x0, v0, R0, Om0), axis=None)

n = 2001
tf = 10.0
t = np.linspace(0, tf, n)
x_des = np.array([0.4*t, 0.4*np.sin(pi*t), 0.6*np.cos(pi*t)])
x_d_des = np.array([0.4*np.ones(n), 0.4*pi*np.cos(pi*t), -0.6*pi*np.sin(pi*t)])
x_2d_des = np.array([np.zeros(n), -0.4*pi**2*np.sin(pi*t), -0.6*pi**2*np.cos(pi*t)])
x_3d_des = np.array([np.zeros(n), -0.4*(pi**3)*np.cos(pi*t), 0.6*(pi**3)*np.sin(pi*t)])
x_4d_des = np.array([np.zeros(n), 0.4*(pi**4)*np.sin(pi*t), 0.6*(pi**4)*np.cos(pi*t)])

b1_dir = np.array([np.cos(pi*t), np.sin(pi*t), np.zeros(n)])
b1_dir_dot = pi * np.array([-np.sin(pi*t), np.cos(pi*t), np.zeros(n)])
b1_dir_ddot = (pi**2) * np.array([-np.cos(pi*t), -np.sin(pi*t), np.zeros(n)])

x = np.empty_like(t)
y = np.empty_like(t)
z = np.empty_like(t)
mat_rot = np.zeros((9, n))

for i in range(1, n):
    x_1 = x_des[0][i]
    x_2 = x_des[1][i]
    x_3 = x_des[2][i]
    x_d = np.array([x_1, x_2, x_3])

    v_1 = x_d_des[0][i]
    v_2 = x_d_des[1][i]
    v_3 = x_d_des[2][i]
    x_d_d = np.array([v_1, v_2, v_3])

    a_1 = x_2d_des[0][i]
    a_2 = x_2d_des[1][i]
    a_3 = x_2d_des[2][i]
    x_2d_d = np.array([a_1, a_2, a_3])

    j_1 = x_3d_des[0][i]
    j_2 = x_3d_des[1][i]
    j_3 = x_3d_des[2][i]
    x_3d_d = np.array([j_1, j_2, j_3])

    s_1 = x_4d_des[0][i]
    s_2 = x_4d_des[1][i]
    s_3 = x_4d_des[2][i]
    x_4d_d = np.array([s_1, s_2, s_3])

    b_1 = b1_dir[0][i]
    b_2 = b1_dir[1][i]
    b_3 = b1_dir[2][i]
    b_dir = np.array([b_1, b_2, b_3])

    b_1_d = b1_dir_dot[0][i]
    b_2_d = b1_dir_dot[1][i]
    b_3_d = b1_dir_dot[2][i]
    b_dir_d = np.array([b_1_d, b_2_d, b_3_d])

    f, M = lee_geom_cont(z0, x_d, x_d_d, x_2d_d, x_3d_d, x_4d_d, b_dir, b_dir_d, t[i-1])
    tspan = [t[i-1], t[i]]
    sol = odeint(model, z0, tspan, args=(f, M,))
    z0 = sol[1]
    x[i] = z0[0]
    y[i] = z0[1]
    mat_rot[:, i - 1] = z0[6:15]
    z[i] = z0[2]

fig1 = plt.figure()
plt.plot(t, x, label="real trajectory along the x axis")
plt.plot(t, x_des[0], label="reference trajectory along the x axis")
plt.xlabel("time [s]")
plt.ylabel("x [m]")
plt.grid()
plt.legend()
plt.show()

fig2 = plt.figure()
plt.plot(t, y, label="real trajectory along the x axis")
plt.plot(t, x_des[1], label="reference trajectory along the y axis")
plt.xlabel("time [s]")
plt.ylabel("y [m]")
plt.grid()
plt.legend()
plt.show()

fig3 = plt.figure()
plt.plot(t, z, label="real trajectory along the x axis")
plt.plot(t, x_des[2], label="reference trajectory along the z axis")
plt.xlabel("time [s]")
plt.ylabel("z [m]")
plt.grid()
plt.legend()
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the trajectory
ax.plot(x, y, z, label="real 3D trajectory")
ax.plot(x_des[0], x_des[1], x_des[2], label="reference 3D trajectory")
# Set labels for the axes
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_zlabel('Z Position')

# Set the title
ax.set_title('3D Position Trajectory')

# Show the plot
plt.show()