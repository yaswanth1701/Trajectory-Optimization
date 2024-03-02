import time as TIME
import argparse
import gym
import numpy as np
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.DynAviary import DynAviary
from gym_pybullet_drones.utils.utils import sync, str2bool


class Quadcopter2D:
    def __init__(self, planning_horizon=200, initial_pos=np.array([[0, 0, 0.5]]), initial_vel=np.zeros(3), initial_orientation=np.zeros(3)):
        '''initialize the quadcopter parameter
        sefl.env: gym environment for the quadcopter (single)

        [action space of quadcopter]
        action[0]: thrust along z axis of the quadcopter(body frame) [first control variable for the 2D case]
        action[1]: torque along the x axis of the quadcopter (for 2D case default=0)
        action[2]: torque along the y axis of the quadcopyer (control variable or optimization variable for the 2D case)
        action[3]: torque along the z axis of the quadcopter (default=0)


        [state space or centre of masss kinemactics in world frame]

        x coordinate and velocity of the quadcopter 
        y coordinate and velocity  of the quadcopter (default=0)
        z coordinate and velocity of the quadcopter
        roll and respective velocity of the quadcopter (default=0)
        pitch and respective velocity of the quadcopter 
        yaw and respective velocityof the quadcopter (default=0)

        '''

        # initialization of environment (for recording the

        self.frequency = 240
        self.env = DynAviary(initial_xyzs=initial_pos, gui=True,
                             record=False, freq=20, user_debug_gui=True)

        obs = self.env.reset()
        self.planning_horizon = planning_horizon

        #  for logging and plots
        actual_states = np.zeros((self.planning_horizon, 6))

        # model parameter
        x, y, z = initial_pos[0]
        vx, vz, omega = initial_vel
        r, p, y = initial_orientation
        self.init = np.array([x, z, p, vx, vz, omega])
        self.L = 397e-4
        self.G = 9.8
        self.M = 0.027
        self.Izz = 1.4e-5

        self.input = np.zeros((planning_horizon, 2))

        # control limit
        self.T2W = 2.25
        self.KF = 3.16e-10

        self.MAX_RPM = np.sqrt((self.T2W*self.G) / (4*self.KF))
        self.env.MAX_THRUST = 4*self.KF*self.MAX_RPM**2

        print("maximum possible  thrust is",  self.env.MAX_THRUST)
        self.MAX_TORQUE = (2*self.L*self.KF*self.MAX_RPM**2)/np.sqrt(2)

        print("maximum possiblle torque is :", self.env.MAX_XY_TORQUE)

        # self.input=np.zeros((planning_horizon,2))

    def store_state(self):
        '''This is for  storing observed states'''
        pass

    def forward_rollout(self, optimal_actions, optimal_states, K):
        '''this for the foward pass of the dynamics to collect states for every iteration may not use this might use RK45'''
        input = optimal_actions

        time = self.planning_horizon
        action = self.env.action_space.sample()
        print("max torque in  the trajectory", max(input[:, 1]))

        actual_states = self.init

        for n in range(input.shape[0]):

            thrust, torque = input[n, :] + K[self.N -
                                             (n+1)] @ np.transpose(actual_states - optimal_states[n, :])
            action['0'] = np.array([thrust, 0, torque, 0])

            print("the  action is  following : ", action['0'])
            obs, reward, done, info = self.env.step(action)
            print("THE pitch is :", obs['0']['state'][8],
                  "THE angular vector :", obs['0']['state'][14])
            self.env.render()
            x, y, z = obs['0']['state'][:3]
            r, p, y = obs['0']['state'][7:10]
            vx, vy, vz = obs['0']['state'][10:13]
            omegax, omegay, omegaz = obs['0']['state'][13:16]
            actual_states = np.array([x, z, p, vx, vz, omegay])
            TIME.sleep(1/20)
        self.env.close()

    def test_env(self):
        action = self.env.action_space.sample()
        for n in range(self.planning_horizon):
            action['0'] = np.array([0, 0, 0, 0])
            obs, reward, done, info = self.env.step(action)
            TIME.sleep(20)


if __name__ == "__main__":
    test = Quadcopter2D(1000)
    test.test_env()
