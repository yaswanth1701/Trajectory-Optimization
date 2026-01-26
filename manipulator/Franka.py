from manipulator_env import *
from scipy.integrate import solve_ivp
import time
import numpy as np

class Trajectory_Optimization(Env):
    def __init__(self, initial_pos, planning_horizon, desired_coordinates, joint_vel=np.zeros(9)):
        super().__init__(initial_pos, desired_coordinates)
        self.dt = 1 / self.frequency
        self.planning_horizon = planning_horizon
        self.iterations = 10
        self.delta = 1e-5
        self.states = np.zeros((planning_horizon, 18))  # 7关节的位置和速度
        self.states[0, :] = np.hstack((self.initial_joint_pos, joint_vel))
        self.actions = np.zeros((planning_horizon, 9))  # 7个关节的扭矩
        for i in range(planning_horizon):
            self.actions[i, :] = self.initial_torque
        self.desired_joint_vel = np.zeros(9)
        self.desired_states = np.hstack((self.desired_joint_pos, self.desired_joint_vel))
        self.K = np.zeros((planning_horizon, 9, 18))
        self.k = np.zeros((planning_horizon, 9, 1))
        self.Q = np.diag([800] * 9 + [200] * 9)
        self.R = np.diag([1] * 9)
        print("Getting initial trajectory states from initial guess of actions (Torques)\n")
        self.get_trajectory()
        initial_cost = self.compute_total_cost()
        print(f"Initial trajectory cost is: {initial_cost}\n")
        print("Starting trajectory optimization...")
        self.run_trajectory_optimization()
        print("Trajectory optimization completed successfully!!!")

    def get_trajectory(self, alpha=1):
        N = self.planning_horizon
        updated_states = np.zeros((self.planning_horizon, 18))
        updated_states[0, :] = self.states[0, :]
        for n in range(self.planning_horizon - 1):
            action = self.actions[n, :] + self.K[N - (n + 1)] @ (updated_states[n, :] - self.states[n, :])  + (alpha * self.k[N - (n + 1)]).ravel()
            self.actions[n, :] = action.copy()
            updated_states[n + 1, :] = self.get_next_state(updated_states[n, :], action)
        self.states = updated_states.copy()

    def get_next_state(self, current_state, action):
        qddot = self.dynamics(current_state, action)
        qdot = current_state[9:]
        next_states = current_state + self.dt * np.hstack((qdot, qddot))
        return next_states

    def dynamics(self, states, action):
        q = states[:9]
        qdot = states[9:]
        qddot_0 = np.zeros(9)
        non_linear_terms, mass_matrix = self.get_dynamics(q, qdot, qddot_0)
        qddot = self.get_joint_acceleration(mass_matrix, non_linear_terms, action)
        return qddot

    def cost_function(self, state, action):
        return (state - self.desired_states).T @ self.Q @ (state - self.desired_states) + action.T @ self.R @ action

    def compute_total_cost(self):
        total_cost = 0
        for n in range(self.planning_horizon):
            total_cost += self.cost_function(self.states[n, :], self.actions[n, :])
        return total_cost

    def get_analytical_derivatives(self, state, action):
        q = state[:7]
        qdot = state[7:]
        torque = action
        pin.computeABADerivatives(self.model, self.data, q, qdot, torque)
        ddq_dq = self.data.ddq_dq
        ddq_dqdot = self.data.ddq_dv
        ddq_du = self.data.Minv
        return ddq_dq, ddq_dqdot, ddq_du

    def gradient_of_dynamics(self, initial_conditions, dim=0):
        var = initial_conditions.copy()
        var[dim] += self.delta
        states = var[:18].copy()
        input = var[18:].copy()
        sol1 = self.get_next_state(states, input)
        var[dim] -= 2 * self.delta
        states = var[:18].copy()
        input = var[18:].copy()
        sol2 = self.get_next_state(states, input)
        return (sol1 - sol2) / (2 * self.delta)

    def compute_gradients(self, state, action, n):
        cost_matrix = np.zeros((27, 27))
        cost_matrix[:18, :18] = self.Q
        cost_matrix[18:, 18:] = self.R
        c_t = 2 * cost_matrix @ np.hstack((state - self.desired_states, action)).T
        C_t = 2 * cost_matrix
        F = np.zeros((18, 27))
        initial_condition = np.hstack((state, action))
        for i in range(27):
            F[:, i] = self.gradient_of_dynamics(initial_condition, i).T
        return c_t, C_t, F

    def line_search(self, old_cost, new_cost):
        alpha = 0
        while new_cost >= old_cost:
            alpha -= 0.2
            if alpha < -1:
                break
            self.get_trajectory(alpha=alpha)
            new_cost = self.compute_total_cost()
            print("Cost in line search is:", new_cost)
        return new_cost

    def run_trajectory_optimization(self):
        sd = 18
        ad = 9
        N = self.planning_horizon
        t1 = time.time()
        for i in range(self.iterations):
            cost_old = self.compute_total_cost()
            c_t, C_t, F = self.compute_gradients(self.states[N - 1, :], self.actions[N - 1, :], N - 1)
            C_x_x = C_t[:sd, :sd].reshape(sd, sd)
            C_x_u = C_t[:sd, sd:].reshape(sd, ad)
            C_u_u = C_t[sd:, sd:].reshape(ad, ad)
            C_u_x = C_t[sd:, :sd].reshape(ad, sd)
            c_x = c_t[:sd].reshape(sd, 1)
            c_u = c_t[sd:].reshape(ad, 1)
            self.K[0] = -np.linalg.inv(C_u_u) @ C_u_x
            self.k[0] = -np.linalg.inv(C_u_u) @ c_u
            V = C_x_x + C_x_u @ self.K[0] + self.K[0].T @ C_u_x + \
                self.K[0].T @ C_u_u @ self.K[0]
            v = c_x + C_x_u @ self.k[0] + self.K[0].T @ c_u + \
                self.K[0].T @ C_u_u @ self.k[0]
            for n in range(1, N):
                c_t, C_t, F = self.compute_gradients(self.states[N - (n + 1)], self.actions[N - (n + 1)], N - (n + 1))
                Q_t = C_t + F.T @ V @ F
                q_t = c_t + (F.T @ v).ravel()
                Q_x_x = Q_t[:sd, :sd].reshape(sd, sd)
                Q_x_u = Q_t[:sd, sd:].reshape(sd, ad)
                Q_u_u = Q_t[sd:, sd:].reshape(ad, ad)
                Q_u_x = Q_t[sd:, :sd].reshape(ad, sd)
                q_x = q_t[:sd].reshape(sd, 1)
                q_u = q_t[sd:].reshape(ad, 1)
                self.K[n] = -np.linalg.inv(Q_u_u) @ Q_u_x
                self.k[n] = -np.linalg.inv(Q_u_u) @ q_u
                V = Q_x_x + Q_x_u @ self.K[n] + self.K[n].T @ Q_u_x + \
                    self.K[n].T @ Q_u_u @ self.K[n]
                v = q_x + Q_x_u @ self.k[n] + self.K[n].T @ q_u + \
                    self.K[n].T @ Q_u_u @ self.k[n]
            self.get_trajectory()
            new_cost = self.compute_total_cost()
            new_cost = self.line_search(cost_old, new_cost)
            print(f"Iteration number {i + 1}: Trajectory cost is {new_cost}")
        t2 = time.time()
        print("Total time taken:", (t2 - t1) * 1e3, "milliseconds")
        error = self.states[N - 1, :] - self.desired_states
        print(f"Final error in states is following: {error}")
        print("Deploying ...")
        self.run(self.actions, TEST=True, planning_horizon=self.planning_horizon,
                 K=self.K, optimal_states=self.states)
        print("Plotting the result \n")
        self.plot()

    def plot(self):
        '''Plotting function for input and state sequence'''
        import matplotlib.pyplot as plt
        with plt.style.context('seaborn-v0_8'):
            plt.figure(figsize=(8, 4))
            plt.plot(self.actions, linestyle='dashdot', label=(
                "joint_torque1", "joint_torque2", "joint_torque3", "joint_torque4", "joint_torque5", "joint_torque6",
                "joint_torque7", "joint_torque8" , "joint_torque9"))
            plt.ylabel("input N")
            plt.xlabel("time")
            plt.legend()
            plt.figure(figsize=(8, 4))
            plt.plot(self.states[:, :9], linestyle='dashdot', label=(
                "joint_pos1", "joint_pos2", "joint_pos3", "joint_pos4", "joint_pos5", "joint_pos6", "joint_pos7","joint_pos8","joint_pos9"))
            plt.plot(self.states[:, 9:], linestyle='dashdot', label=(
                "joint_vel1", "joint_vel2", "joint_vel3", "joint_vel4", "joint_vel5", "joint_vel6", "joint_vel7","joint_vel8","joint_vel9"))
            plt.ylabel("states")
            plt.xlabel("time")
            plt.legend()
            plt.show()

if __name__ == "__main__":
    initial_pos = np.array([0.8, -0.1, 0.5])  # 设定适用于7关节的初始位置
    planning_horizon = 800
    desired_coordinates = np.array([0.4, 0.2, 0.5])  # 设定适用于7关节的目标位置
    test = Trajectory_Optimization(initial_pos, planning_horizon, desired_coordinates)
