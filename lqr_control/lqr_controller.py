import numpy as np
import math
import time
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.integrate import odeint
import scipy

class LQRTrajectoryControl:
    def __init__(self, v_average, path, dt):
        self.path = path
        self.start = (path[0][0], path[0][1], path[0][2])
        self.goal = (path[-1][0]*10, path[-1][1]*10, path[-1][2]*10)
        self.dist_path = self.calculate_distance_path()
        self.v_average = v_average
        self.T = int(self.dist_path / self.v_average)
        self.dt = dt

        self.x_vel, self.y_vel, self.z_vel, self.x_pos, self.y_pos, self.z_pos = self.Cubic_spline(self.path, self.T, self.dt)

        self.fig, self.ax = self.plot_trajectory()
        self.distance = 0
        self.angle_theta = []
        self.Ax, self.Bx, self.Ay, self.By, self.Az, self.Bz, self.Ayaw, self.Byaw = self.initialize_matrices()
        self.Ks = self.lqr_solution()
        self.X0 = np.zeros(12)
        self.signalyaw = np.zeros_like(self.x_pos) #The yaw angle is not used in this simulation
        self.t = np.arange(0., self.T, self.dt)
        self.x_linear = odeint(self.cl_linear, self.X0, self.t, args=(self.u,))

        fig = plt.figure()
        track = fig.add_subplot(projection="3d")
        track.text(self.x_pos[0], self.y_pos[0], self.z_pos[0], "start", color='red')
        track.text(self.x_pos[-1], self.y_pos[-1], self.z_pos[-1], "finish", color='red')
        track.plot(self.x_linear[:, 0], self.x_linear[:, 2], self.x_linear[:, 4], color="r", label="real position trajectory")
        track.plot(self.x_pos, self.y_pos, self.z_pos, color="b", label="simulated position trajectory")
        track.set_xlim(0, 500)
        track.set_ylim(0, 850)
        track.set_zlim(0, 270)
        track.legend()
        track.set_title('The Generated LQR Position Trajectory')
        plt.show()

        fig1 = plt.figure()
        plt.plot(self.x_linear[:,1], label="real velocities along x-axis")
        plt.xlabel("time [s]")
        plt.ylabel("v [m/s]")
        plt.grid()
        plt.legend()
        plt.show()

        fig2 = plt.figure()
        plt.plot(self.x_linear[:,3], label="real velocities along y-axis")
        plt.xlabel("time [s]")
        plt.ylabel("v [m/s]")
        plt.grid()
        plt.legend()
        plt.show()

    def calculate_distance_path(self):
        dim = np.shape(self.path)[0]
        dist_path = 0
        for i in range(1, dim):
            a = (self.path[i][0] - self.path[i-1][0]) * 10
            b = (self.path[i][1] - self.path[i-1][1]) * 10
            c = (self.path[i][2] - self.path[i-1][2]) * 10
            dist = math.sqrt(a ** 2 + b ** 2 + c ** 2)
            dist_path += dist
        return dist_path

    def Cubic_spline(self, waypoints, T, dt):
        n = np.shape(waypoints)
        dim = n[0]
        t = np.linspace(0, T, dim)
        N_points = math.floor(T / dt)
        x = [waypoints[0][0]]
        y = [waypoints[0][1]]
        z = [waypoints[0][2]]

        for i in range(1, dim):
            a = waypoints[i][0] * 10
            b = waypoints[i][1] * 10
            c = waypoints[i][2] * 10
            x.append(a)
            y.append(b)
            z.append(c)

        t_new = np.linspace(min(t), max(t), N_points)
        f1 = CubicSpline(t, x, bc_type='natural')
        f2 = CubicSpline(t, y, bc_type='natural')
        f3 = CubicSpline(t, z, bc_type='natural')
        x_pos = f1(t_new)
        y_pos = f2(t_new)
        z_pos = f3(t_new)

        dim_x = np.shape(x_pos)
        dim_x = dim_x[0]

        x_dif = []
        y_dif = []
        z_dif = []
        angle_theta = []

        vel_x = [0.0]
        vel_y = [0.0]
        vel_z = [0.0]

        for i in range(1, dim_x):
            dx = (x_pos[i] - x_pos[i - 1]) 
            dy = (y_pos[i] - y_pos[i - 1]) 
            dz = (z_pos[i] - z_pos[i - 1])

            x_dif.append(dx)
            y_dif.append(dy)
            z_dif.append(dy)

            dist = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

            theta = math.degrees(math.atan2(dy, dx))
            phi = math.degrees(math.atan2(dz, dist))

            angle_theta.append(theta)

            vx = (self.v_average * math.cos(math.radians(theta)))
            vy = (self.v_average * math.sin(math.radians(theta)))
            vz = (self.v_average * math.sin(math.radians(phi)))

            vel_x.append(vx)
            vel_y.append(vy)
            vel_z.append(vz)

        x_vel = vel_x
        y_vel = vel_y
        z_vel = vel_z

        return x_vel, y_vel, z_vel, x_pos, y_pos, z_pos

    def plot_trajectory(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        line = ax.plot3D(self.x_pos, self.y_pos, self.z_pos, 'blue', label='Drone Path')
        ax.set_title('3D Drone Path Trajectory')
        ax.scatter(*self.start, c='green', s=100, label='Start')
        ax.scatter(*self.goal, c='red', s=100, label='Goal')
        ax.set_xlim(0, 500)
        ax.set_ylim(0, 850)
        ax.set_zlim(0, 270)
        ax.legend()
        plt.show()
        return fig, ax

    def initialize_matrices(self):
        Ax = np.array([[0.0, 1.0, 0.0, 0.0],
                       [0.0, 0.0, 9.81, 0.0], # g = 9.81 N.m²/kg²
                       [0.0, 0.0, 0.0, 1.0],
                       [0.0, 0.0, 0.0, 0.0]])
        Bx = np.array([[0.0],
                       [0.0],
                       [0.0],
                       [1 / 0.00678]]) # Ix = Iy = 0.00678 kg.m²
        Ay = np.array([[0.0, 1.0, 0.0, 0.0],
                       [0.0, 0.0, -9.81, 0.0], # g = 9.81 N.m²/kg²
                       [0.0, 0.0, 0.0, 1.0],
                       [0.0, 0.0, 0.0, 0.0]])
        By = np.array([[0.0],
                       [0.0],
                       [0.0],
                       [1 / 0.00678]]) # Ix = Iy = 0.00678 kg.m²
        Az = np.array([[0.0, 1.0],
                       [0.0, 0.0]])
        Bz = np.array([[0.0],
                       [1 / 0.08]]) # m = 0.08 kg
        Ayaw = np.array([[0.0, 1.0],
                         [0.0, 0.0]])
        Byaw = np.array([[0.0],
                         [1 / 0.01313]]) #Iz = 0.01313 kg.m²
        return Ax, Bx, Ay, By, Az, Bz, Ayaw, Byaw

    def lqr_solution(self):
        Ks = []
        for A, B in ((self.Ax, self.Bx), (self.Ay, self.By), (self.Az, self.Bz), (self.Ayaw, self.Byaw)):
            n = A.shape[0]
            m = B.shape[1]
            Q = np.eye(n)
            Q[0, 0] = 100000000
            Q[1, 1] = 10
            R = np.diag([1., ])
            K, _, _ = self.lqr(A, B, Q, R)
            Ks.append(K)
        return Ks

    def lqr(self, A, B, Q, R):
        X = np.matrix(scipy.linalg.solve_continuous_are(A, B, Q, R))
        K = np.matrix(scipy.linalg.inv(R) * (B.T * X))
        eigVals, eigVecs = scipy.linalg.eig(A - B * K)
        return np.asarray(K), np.asarray(X), np.asarray(eigVals)

    def cl_linear(self, x, t, u):
        x = np.array(x)
        X, Y, Z, Yaw = x[[0, 1, 8, 9]], x[[2, 3, 6, 7]], x[[4, 5]], x[[10, 11]]
        UZ, UY, UX, UYaw = u(x, t).reshape(-1).tolist()
        dot_X = self.Ax.dot(X) + (self.Bx * UX).reshape(-1)
        dot_Y = self.Ay.dot(Y) + (self.By * UY).reshape(-1)
        dot_Z = self.Az.dot(Z) + (self.Bz * UZ).reshape(-1)
        dot_Yaw = self.Ayaw.dot(Yaw) + (self.Byaw * UYaw).reshape(-1)
        dot_x = np.concatenate([dot_X[[0, 1]], dot_Y[[0, 1]], dot_Z, dot_Y[[2, 3]], dot_X[[2, 3]], dot_Yaw])
        return dot_x

    def u(self, x, _t):
        dis = _t - self.t
        dis[dis < 0] = np.inf
        idx = dis.argmin()
        UX = self.Ks[0].dot(np.array([self.x_pos[idx], self.x_vel[idx], 0, 0]) - x[[0, 1, 8, 9]])[0]
        UY = self.Ks[1].dot(np.array([self.y_pos[idx], self.y_vel[idx], 0, 0]) - x[[2, 3, 6, 7]])[0]
        UZ = self.Ks[2].dot(np.array([self.z_pos[idx], self.z_vel[idx]]) - x[[4, 5]])[0]
        UYaw = self.Ks[3].dot(np.array([self.signalyaw[idx], 0]) - x[[10, 11]])[0]
        return np.array([UZ, UY, UX, UYaw])

def main():
    path = [(0, 0, 0), (2, 2, 2), (4, 4, 4), (6, 6, 6), 
            (8, 8, 8), (10, 10, 10), (12, 12, 10), (14, 13, 10)]

    # Initial parameters
    v_average = 35    # average speed of the drone in cm/s
    dt = 0.04         # f = 25 Hz: frequency used to send the velocity commands

    traj_control = LQRTrajectoryControl(v_average, path, dt)
    x_dynamics = traj_control.x_linear
    vel_x = x_dynamics[:, 1]
    vel_y = x_dynamics[:, 3]
    vel_z = x_dynamics[:, 5]
    print("The velocity profile is", (vel_x, vel_y, vel_z))

if __name__ == '__main__':
    main()