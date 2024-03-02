import time as TIME
import argparse
import gym
import numpy as np
from scipy import linalg
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.DynAviary import DynAviary
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import DroneModel, Physics


class Quadcopter3D:
    def __init__(self, initial_pos=np.array([[1, 2, 0.3]]), initial_vel=np.zeros(3), initial_orientation=np.array([0, 0, 0])):
        '''initialize the quadcopter parameter
        sefl.env: gym environment for the quadcopter (single)
        THIS ENVIRONMENT IS FOR APPLYING ILQR FOR SINGLE 3D QUADCOPTER USING QUATERNION DYNAMICS
        STATE
        - state:
        1) x position (world frame)
        2) y position (world frame)
        3) z position (world frame)
        4) x quaternion
        5) y quaternion
        6) z quaternion
        7) w quaternion
        ## body frame velcities
        8) velocity_x
        9) velcity_y
        10) velocity_z
        11) omega_x
        12) omega_y
        13) omega_z
        '''

        # initialization of environment (for recording the

        self.frequency = 240
        self.env = DynAviary(drone_model=DroneModel.HB, initial_xyzs=initial_pos.reshape(1, 3), initial_rpys=initial_orientation.reshape(1, 3), gui=True,
                             record=False, freq=20, user_debug_gui=True)
        state = self.env._getDroneStateVector(0)
        print(f"the state is {state}")

        obs = self.env.reset()
        self.sim_time = 2000

        #  for logging and plots

        # model parameter
        x, y, z = initial_pos[0]
        vx, vz, omega = initial_vel
        r, p, y = initial_orientation
        self.init = np.array(
            [1, 2, 0.3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape(13, 1)
        self.L = 0.175
        self.G = 9.8
        self.M = 0.5
        self.Ixx = 23e-4
        self.Iyy = 23e-4
        self.Izz = 4e-3
        self.J = np.diag([self.Ixx, self.Iyy, self.Izz])
        self.J_inv = np.linalg.inv(self.J)
        self.delt_t = 0.05
        # conjugate matrix
        self.T = np.diag([1, -1, -1, -1])
        # hat map
        self.H = np.array([[0, 0, 0],
                           [1, 0, 0],
                           [0, 1, 0],
                           [0, 0, 1]])
        self.Q = np.diag(np.array([1, 1, 10, 1, 1, 1, 1, 1, 1, 1, 1, 1]))
        self.R = np.diag(np.array([200, 200, 200, 200]))

        # print(f"the interia matrix is {self.J}")
        # print(
        #     f"the rotation matrix is \n {self.step(np.array([1,0,0,np.sqrt(0.5),np.sqrt(0.5),0,0,0,0,0,0,0,0]).reshape(13,1),np.zeros((4,1)))}")
        self.hover_state()
        self.cal_A_B()

        print(f"\n {self.red_A}\n")
        print(f"\n {self.red_B}\n")
        self.lqr_gains()
        print(self.K)

    def hover_state(self):

        self.r_hover = np.array([0, 0, 1]).reshape(3, 1)
        self.w_hover = np.array([1, 0, 0, 0]).reshape(4, 1)
        self.v_hover = np.zeros((3, 1))
        self.omega_hover = np.zeros((3, 1))
        self.u_hover = np.array([self.M*self.G, 0, 0, 0]).reshape(4, 1)
        self.hover_state = np.zeros((13, 1))
        self.hover_state[:3] = self.r_hover
        self.hover_state[3:7] = self.w_hover
        self.hover_state[7:10] = self.v_hover
        self.hover_state[10:] = self.omega_hover

    def grad_dyn_state(self, num):

        delt_x = 1e-5
        x_pls = self.hover_state.copy()
        x_pls[num] = x_pls[num] + delt_x
        x_plus = self.quad_dynamics(x_pls, self.u_hover)

        x_min = self.hover_state.copy()
        x_min[num] = x_min[num] - delt_x
        x_minus = self.quad_dynamics(x_min, self.u_hover)

        grad_x = (x_plus - x_minus)/(2*delt_x)

        return grad_x.ravel()

    def grad_dyn_input(self, num):

        delt_x = 1e-5
        u_plus = self.u_hover.copy()
        u_plus[num] = u_plus[num] + delt_x
        x_plus = self.quad_dynamics(self.hover_state, u_plus)

        u_minus = self.u_hover.copy()
        u_minus[num] = u_minus[num] - delt_x
        x_minus = self.quad_dynamics(self.hover_state, u_minus)

        grad_u = (x_plus - x_minus)/(2*delt_x)

        return grad_u.ravel()

    def cal_A_B(self):

        self.A = np.zeros((13, 13))
        self.B = np.zeros((13, 4))
        # for states
        for i in range(13):
            self.A[:, i] = self.grad_dyn_state(i)

        for i in range(4):
            self.B[:, i] = self.grad_dyn_input(i)

        self.red_A = self.E(self.w_hover).T @ self.A @ self.E(self.w_hover)
        self.red_B = self.E(self.w_hover).T @ self.B

    def qtorot(self, q):
        return self.H.T @ self.T @ self.L_q(q) @ self.T @ self.L_q(q) @ self.H

    def L_q(self, q):
        s = q[0]
        v = q[1:]
        L1 = np.hstack((np.array([s]), -v.T))
        L2 = np.hstack((v, s * np.identity(3) + self.hat(v.ravel())))
        L = np.vstack((L1, L2))
        return L

    def hat(self, v):
        return np.array([[0, - v[2], v[1]],
                         [v[2], 0, - v[0]],
                         [-v[1], v[0], 0]])

    def G_q(self, q):
        return self.L_q(q) @ self.H

    def axtoq(self, phi):
        return (1/(np.sqrt(1 + phi.T @ phi)))*np.array([1, phi[0][0], phi[1][0], phi[2][0]]).reshape(4, 1)

    def qtoax(self, q):
        return q[1:]/q[0]

    def E(self, q):

        I3 = np.identity(3)
        I6 = np.identity(6)

        return np.block([[I3, np.zeros((3, 9))],
                         [np.zeros((4, 3)), self.G_q(q), np.zeros((4, 6))],
                         [np.zeros((6, 6)), I6]])

    def step(self, curr_state, input):
        # simple euler integration
        # getting rate of state
        state_dot = self.quad_dynamics(curr_state, input)
        # euler integrationstep
        next_state = curr_state + self.delt_t * state_dot
        # normalizing the quaternion
        next_state[3:7] = next_state[3:7]/np.linalg.norm(next_state[3:7])

        return next_state

    def quad_dynamics(self, states, input):

        r = states[:3]
        q = states[3:7]/np.linalg.norm(states[3:7])
        v = states[7:10]
        omega = states[10:]
        thrust = input[0]
        torque = input[1:]
        # kinematics part
        r_dot = self.qtorot(q) @ v
        q_dot = 0.5*self.L_q(q) @ self.H @ omega

        # dynamics part
        self.v_dot = self.qtorot(q).T @ np.array([[0], [0], [-self.M*self.G]]) + np.array([[0], [0], thrust])\
            - self.hat(omega.ravel()) @ v

        omega_dot = self.J_inv @ (torque -
                                  self.hat(omega.ravel()) @ self.J @ omega)
        # x_dot
        state_dot = np.block([[r_dot],
                              [q_dot],
                              [self.v_dot],
                              [omega_dot]])
        return state_dot

    def lqr_gains(self):
        S = linalg.solve_continuous_are(
            self.red_A, self.red_B, self.Q, self.R)
        self.K = np.linalg.inv(self.R)@np.transpose(self.red_B)@S

        return self.K

    def cal_delta_state(self, actual_state):

        delta_state = np.zeros((12, 1))
        q = actual_state[3:7]
        phi = self.qtoax(self.L_q(self.w_hover).T @ q)

        delta_state[:3] = actual_state[:3] - self.hover_state[:3]
        delta_state[3:6] = phi
        delta_state[6:9] = actual_state[7:10] - self.hover_state[7:10]
        delta_state[9:] = actual_state[10:] - self.hover_state[10:]

        return delta_state

    def rollout(self):
        '''this for the foward pass of the dynamics to collect states for every iteration may not use this might use RK45'''
        actual_states = self.init
        action = self.env.action_space.sample()

        while True:
            delta_state = self.cal_delta_state(actual_states)
            print(f"the delt state {delta_state}")
            u = self.u_hover - self.K @ delta_state
            print(f"the input is {u}")
            thrust, torque_x, torque_y, torque_z = u.ravel()
            print(f"{thrust, torque_x, torque_y, torque_z}")
            action['0'] = np.array([thrust, torque_x, torque_y, torque_z])
            obs, reward, done, info = self.env.step(action)
            self.env.render()
            x, y, z = obs['0']['state'][:3]
            x_r, y_r, z_r, w_r = obs['0']['state'][3:7]
            vx, vy, vz = obs['0']['state'][10:13]
            omegax, omegay, omegaz = obs['0']['state'][13:16]
            print(f"orientation {x_r, y_r, z_r, w_r}")

            q = np.array([w_r, x_r, y_r, z_r]).reshape(4, 1)
            # body frame rate linear and angular
            Q = self.qtorot(q).T
            v_b = Q @ np.array([vx, vy, vz]).reshape(3, 1)
            omega_b = Q @ np.array([omegax, omegay, omegaz]).reshape(3, 1)
            v_bx, v_by, v_bz = v_b.ravel()
            omega_bx, omega_by, omega_bz = omega_b.ravel()

            actual_states = np.array(
                [x, y, z, w_r, x_r, y_r, z_r, v_bx, v_by, v_bz, omega_bx, omega_by, omega_bz]).reshape(13, 1)
            TIME.sleep(1/240)
        self.env.close()


if __name__ == "__main__":
    test = Quadcopter3D()
    test.rollout()
