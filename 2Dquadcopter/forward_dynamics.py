from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np
from quadcopter2Denv import Quadcopter2D
import time


class dynamics_solver(Quadcopter2D):

    def __init__(self):
        '''This class has methods to solve the dynamics using RK45 method for forward_rollout and back_rollout(gradients calculations
        attributes:
        self.dt: step size for integration is set to  1/240 as pybullet uses the same step size
        self.dynamics:  function which returns the differential equation for solve_ivp (scipy.integrate)'''

        super().__init__()
        self.dt = 0.05

        self.stdim = 6

        self.actdim = 2

        self.totdim = self.stdim + self.actdim

        self.iterations = 20

        self.delta = 1e-5

        self.dynamics = lambda T, X, thrust, torque: [X[3],                               X[4],                          X[5],
                                                      np.sin(X[2])*thrust/self.M, (np.cos(X[2])*thrust/self.M)-self.G, torque/self.Izz]

        self.Q = np.array([[10, 0, 0, 0, 0, 0],
                           [0, 10, 0, 0, 0, 0],
                           [0, 0, 10, 0, 0, 0],
                           [0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 0, 10]])

        self.R = np.array([[1, 0],
                           [0, 1]])

        self.K = np.zeros((self.planning_horizon, self.actdim, self.stdim))
        self.k = np.zeros((self.planning_horizon, self.actdim, 1))

        self.input = np.zeros((self.planning_horizon, self.actdim))
        self.input[:, 0] = np.ones((self.planning_horizon))*self.M*self.G

        self.radius = 0.5

        self.updated_states = np.zeros((self.planning_horizon, 6))
        self.updated_states[0, :] = self.init
        self.states = self.updated_states.copy()

        # self.x = self.radius*np.cos(np.linspace(0,2*np.pi,num  =  self.planning_horizon))

        # self.z = self.radius*np.sin(np.linspace(0,2*np.pi,num  =  self.planning_horizon)) + self.init[1]

        self.x = np.ones(self.planning_horizon)*0.5
        self.z = np.ones(self.planning_horizon)*0.8

        self.tarstate = np.zeros((self.planning_horizon, 2))
        self.tarstate[:, 0] = self.x

        self.tarstate[:, 1] = self.z

    def update_trajectory(self, alpha=1):
        '''given the initial state and input trajectory computes states for the entrie horizon



        Parameters:
        input: input to the systemqtoQ numpy array of size  planning  horizon x 2
        return:
        states: states of the system numpy array of size  planning horizon x 6'''

        self.N = self.planning_horizon

        for n in range(self.planning_horizon-1):

            action = self.input[n, :] + self.K[self.N-(n+1)]@(np.transpose(
                self.updated_states[n, :]-self.states[n, :])) + alpha*self.k[self.N-(n+1)].ravel()
            self.input[n, :] = action.copy()

            sol = solve_ivp(self.dynamics, [
                            0, self.dt], self.updated_states[n, :], first_step=self.dt, args=action)
            Y = sol.y
            T = sol.t

            Y[2, 1] = np.arctan2(np.sin(Y[2, 1]), np.cos(Y[2, 1]))
            self.updated_states[n+1, :] = Y[:, 1].T

        self.states = self.updated_states.copy()

        # plt.plot(self.states[:,2])

        # plt.legend()
        # plt.show()

    def gradient_of_dynamics(self, initial_condtions, dim=0):
        ''' using finite difference method computes the differentiation of whole vector valued function with respect  to the  passed variable

        Parameter:

        initial_conditon : value at  which the  derivative is required (vector of state and input variables  stacked  horizontally at required  time step)


        dim  : Is  the variable with which we  want to compute the derivative (integer)


        Return :

        returns  a  row   vector (derivative value)


        '''

        var = initial_condtions.copy()
        var[dim] = var[dim] + self.delta

        states = var[:self.stdim].copy()

        input = var[self.stdim:self.totdim].copy()

        sol = solve_ivp(self.dynamics, [0, self.dt],
                        states, first_step=self.dt, args=input)

        Yplus = sol.y

        var[dim] = var[dim] - 2*self.delta

        states = var[:self.stdim].copy()

        input = var[self.stdim:self.totdim].copy()

        sol = solve_ivp(self.dynamics, [0, self.dt],
                        states, first_step=self.dt, args=input)

        Yminus = sol.y

        return (Yplus[:, 1] - Yminus[:, 1]) / (2*self.delta)

    def get_gradient(self, states, input, n):

        Q = self.Q

        tarstate = self.tarstate

        R = self.R

        c_t = 2*np.array([[Q[0, 0]*(states[0] - tarstate[n, 0])],
                          [Q[1, 1]*(states[1] - tarstate[n, 1])],
                          [Q[2, 2]*(states[2])],
                          [Q[3, 3]*(states[3])],
                          [Q[4, 4]*(states[4])],
                          [Q[5, 5]*(states[5])],
                          [R[0, 0]*(input[0])],
                          [R[1, 1]*input[1]]])

        C_t = 2 * np.array([[Q[0, 0], 0, 0, 0, 0, 0, 0, 0],

                           [0, Q[1, 1], 0, 0, 0, 0, 0, 0],

                           [0, 0, Q[2, 2], 0, 0, 0, 0, 0],

                           [0, 0, 0, Q[3, 3], 0, 0, 0, 0],

                           [0, 0, 0, 0, Q[4, 4], 0, 0, 0],

                           [0, 0, 0, 0, 0, Q[5, 5], 0, 0],

                           [0, 0, 0, 0, 0, 0, R[0, 0], 0],

                           [0, 0, 0, 0, 0, 0, 0, R[1, 1]]])

        initial_condition = np.hstack((states, input))
        # initial_condition = jnp.zeros(8)
        # F = jacfwd(self.gradient_of_dynamics)(jnp.array(initial_condition))
        F = np.zeros((6, 8))

        # print("here is the new gradient",F )

        for i in range(self.totdim):
            F[:, i] = self.gradient_of_dynamics(initial_condition, i).T

        return c_t, C_t, F

    def compute_cost(self, state, input, i):

        return (state - np.array([self.tarstate[i, 0], self.tarstate[i, 1], 0, 0, 0, 0])).T  @ self.Q @ (state - np.array([self.tarstate[i, 0], self.tarstate[i, 1], 0, 0, 0, 0])) + \
            (input - np.array([self.M*self.G, 0])
             ).T  @ self.R @ (input - np.array([self.M*self.G, 0]))

    def compute_trajectory_cost(self):

        total_cost = 0

        for i in range(self.planning_horizon):

            total_cost += self.compute_cost(self.states[i], self.input[i], i)

        return total_cost

    def line_search(self, old_cost, new_cost):
        alpha = 0
        while new_cost >= old_cost:

            alpha = alpha - 0.2
            if alpha < -1:
                break
            self.update_trajectory(alpha=alpha)
            new_cost = self.compute_trajectory_cost()
            print("cost in line search is :", new_cost)

    def boxddp(self):

        sd = self.stdim
        ad = self.actdim

        t1 = time.time()

        for i in range(self.iterations):

            cost_old = self.compute_trajectory_cost()

            c_t, C_t, F = self.get_gradient(self.states[self.planning_horizon-1, :],
                                            self.input[self.planning_horizon-1, :], self.planning_horizon-1)

            C_x_x = C_t[:sd, :sd].reshape(sd, sd)
            C_x_u = C_t[:sd, sd:].reshape(sd, ad)

            C_u_u = C_t[sd:, sd:].reshape(ad, ad)
            C_u_x = C_t[sd:, :sd].reshape(ad, sd)

            c_x = c_t[:sd].reshape(sd, 1)
            c_u = c_t[sd:].reshape(ad, 1)

            self.K[0] = -np.linalg.inv(C_u_u) @ C_u_x
            self.k[0] = -np.linalg.inv(C_u_u) @ c_u

            V = C_x_x + C_x_u @ self.K[0] + np.transpose(self.K[0]) @ C_u_x + \
                np.transpose(self.K[0]) @ C_u_u @ self.K[0]

            v = c_x + C_x_u @ self.k[0] + np.transpose(self.K[0]) @ c_u + \
                np.transpose(self.K[0]) @ C_u_u @ self.k[0]

            for n in range(1, self.N):

                c_t, C_t, F = self.get_gradient(self.states[self.planning_horizon-(n+1)],
                                                self.input[self.planning_horizon-(n+1)], self.planning_horizon-(n+1))

                Q_t = C_t + np.transpose(F) @ V @ F
                q_t = c_t + np.transpose(F) @ v

                Q_x_x = Q_t[:sd, :sd].reshape(sd, sd)
                Q_x_u = Q_t[:sd, sd:].reshape(sd, ad)

                Q_u_u = Q_t[sd:, sd:].reshape(ad, ad)
                Q_u_x = Q_t[sd:, :sd].reshape(ad, sd)

                q_x = q_t[:sd].reshape(sd, 1)
                q_u = q_t[sd:].reshape(ad, 1)

                self.K[n] = -np.linalg.inv(Q_u_u) @ Q_u_x
                self.k[n] = -np.linalg.inv(Q_u_u) @ q_u

                V = Q_x_x + Q_x_u @ self.K[n] + np.transpose(self.K[n]) @ Q_u_x + \
                    np.transpose(self.K[n]) @ Q_u_u @ self.K[n]

                v = q_x + Q_x_u @ self.k[n] + np.transpose(self.K[n]) @ q_u \
                    + np.transpose(self.K[n]) @ Q_u_u @ self.k[n]

            self.update_trajectory()

            new_cost = self.compute_trajectory_cost()

            self.line_search(cost_old, new_cost)

            print("iteration number : ", i+1,
                  "corresponding trajectory cost : ", new_cost)

        t2 = time.time()
        print("total  time  taken:  ", (t2 - t1)*1e+3, "millisecond")

        self.plot()
        self.forward_rollout(self.input, self.states, self.K)

    def plot(self):
        '''plotting function for input and state sequence'''
        with plt.style.context('seaborn-v0_8'):
            plt.figure(figsize=(8, 4))
            plt.plot(self.input, linestyle='dashdot',
                     label=("thurst", "torque"))
            plt.ylabel("input N")
            plt.xlabel("time")
            plt.legend()
            plt.figure(figsize=(8, 4))
            plt.plot(self.states, linestyle='dashdot', label=(
                'x', "z", "theta", "vx", "vz", "omega"))
            plt.ylabel("states")
            plt.xlabel("time")
            plt.legend()
        plt.show()


if __name__ == "__main__":
    test = dynamics_solver()

    # faster_function = jit(test.update_trajectory)
    # faster_function()

    test.update_trajectory()
    test.boxddp()
