import matplotlib.pyplot as plt
import numpy as np
import gym
from torch.autograd.functional import jacobian
from torch import tensor
import time
from gym import logger



class Cartpole:
    def __init__(self):

        '''all the  neccessary parameters  for the system and  
        lqr control and cost  parameters'''

        # system parameters
        self.m =0.1
        self.g = 9.8
        self.l = 0.5
        self.M = 1
        self.total_mass = 1.1
        self.ml = self.m*self.l

        # time step size
        self.dt = 0.02
 
        # planning horizon
        self.N = 220

        # loading environment
        self.env = gym.make('CartPole-v1')
        self.testenv = gym.make('CartPole-v1', render_mode="rgb_array")
        

        # random input sequence
        self.u = np.ones((self.N,2))

        # lqr gains
        self.K = np.zeros((self.N, 2,4))
        self.k = np.zeros((self.N, 2,1))


        # random state sequence
        self.x = np.zeros((self.N, 4))
        

        #  updated states  sequence
        self.updatedx = np.zeros((self.N, 4))
        self.x[0, :] = [0,0,np.pi, 0]
        self.updatedx[0,:] = [0, 0 ,np.pi,0]

        # dynamics 
        self.f = np.zeros((4, 1))
        # cost  function weights

        
        # cofficient for each system variable
        self.Q = np.array([[20,0,0,0],[0,1,0,0],[0,0,20,0],[0,0,0,1]])
        self.R = np.array([[1,0],[0,1]])
    


    def forward(self):
            '''computing initial states from random input sequence'''
            self.env.reset()
            for n  in range(0,self.N):
                state=self.env.state

                self.updatedx[n,:] = np.array([state[0],state[1],state[2]
                                             ,state[3]])

                action=self.u[n]

                observation, reward, terminated, truncated, info = self.env.step(action[0])
            self.x=self.updatedx.copy()

            self.env.close()



    def update_traj(self,alpha=1):
        '''for updating the states and input sequence every iteration'''


        self.env.reset()

        for n in range(0,self.N):
            self.updatedx[n,:]= np.array([self.env.state[0],self.env.state[1],self.env.state[2]
                                             ,self.env.state[3]]).ravel()

            state=np.array([self.env.state[0],self.env.state[1],self.env.state[2]
                                             ,self.env.state[3]]).ravel()

            action = self.u[n]+self.K[self.N-(n+1)]@(np.transpose(state-self.x[n]))+alpha*self.k[self.N-(n+1)].ravel()


            self.u[n]=action.copy()

            observation, reward, terminated, truncated, info = self.env.step(action[0])

        self.x=self.updatedx.copy()



    def statespacemodel(self):

        '''temp = (f + ml * omega**2 * sin(theta)) / self.total_mass
        thetaacc = (g * sin(theta) - cos(theta) * temp) / (self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass'''
        pass



    def compute_gradeint(self,theta,omega,force):
        ''' this method is used for computing the gradient of the dynamics vector'''
        f = force
        ml = self.ml
        total_mass = self.total_mass
        g = self.g
        l=self.l
        m=self.m


        temp = (f+ml*(omega**2)*np.sin(theta))/total_mass
        alpha =  (g*np.sin(theta)- np.cos(theta)*temp)/(l*(4/3-m*np.cos(theta)**2/total_mass))
        acceleration = temp - ml * alpha * np.cos(theta) / total_mass
        
        return alpha,acceleration


    def compute_traj_cost(self):
        '''Computing the cost for every state and input trajectory'''
        
        Q = self.Q
        R = self.R

        trajectory_cost = 0


        for n in range(self.N):
            trajectory_cost+=self.x[n,:] @ Q @ np.transpose(self.x[n,:]) + self.u[n,:] @ R @ np.transpose(self.u[n,:])
        return trajectory_cost



    def gradients(self,displacement,velocity,theta,omega,force,force2):
        '''' calculating gradients of system dynamics and cost function
        dyngrad: is jacobian of the system dynamics
        self.c : is the first derivative of the cost function
        self.C : is the second derivative of the cost function or jacobian of the self.c
        '''
        m=self.m 
        g=self.g 
        l=self.l 
        M=self.M
        total_mass=self.total_mass
        ml=self.ml
        dt=self.dt
        dx=1e-2


        ### gradient computation using finite
        #difference method

        gradtx = (self.compute_gradeint(theta + dx, omega, force)[1]-self.compute_gradeint(theta - dx, omega, force)[1])/(2*dx)
        gradtt = (self.compute_gradeint(theta + dx, omega, force)[0]-self.compute_gradeint(theta - dx, omega, force)[0])/(2*dx)


        gradfx  = (self.compute_gradeint(theta , omega, force+dx)[1]-self.compute_gradeint(theta, omega, force - dx)[1])/(2*dx)
        gradft  = (self.compute_gradeint(theta , omega, force+dx)[0]-self.compute_gradeint(theta, omega, force - dx)[0])/(2*dx)

        gradox  =  (self.compute_gradeint(theta , omega+dx, force)[1]-self.compute_gradeint(theta, omega- dx, force )[1])/(2*dx)
        gradot  =  (self.compute_gradeint(theta , omega+dx, force)[0]-self.compute_gradeint(theta, omega- dx, force )[0])/(2*dx)

        # the jacobian matrix   

        dyngrad = np.array([[1   ,   dt   ,   0  ,      0  ,  0        , 0],  
                            [0   ,   1    ,  dt*gradtx ,   dt*gradox ,  dt*gradfx, 0],
                            [0   ,   0    ,    1       ,dt ,  0        , 0],
                            [0   ,   0    ,  dt*gradtt  ,1+dt*gradot , dt*gradft, 0]])
        f = np.zeros((4,1))
       
        #gradient of cost function

        self.c = 2*np.array([[self.Q[0,0] * displacement],
                             [self.Q[1,1] * velocity],
                             [self.Q[2,2] * theta], 
                             [self.Q[3,3] * omega],
                             [self.R[0,0] * force],
                             [self.R[1,1] * force2]])

        # jocabian of gradient vector self.c

        self.C= 2*np.array([[self.Q[0,0],0,0,0,0,0],

                            [0,self.Q[1,1],0,0,0,0],

                            [0,0,self.Q[2,2],0,0,0],

                            [0,0,0,self.Q[3,3],0,0],

                            [0,0,0,0,self.R[0,0],0],

                            [0,0,0,0,0,self.R[1,1]]])

        return dyngrad,self.C,self.c,f



    def ilqr(self):
        '''  here the ilqr algorithm is implemeneted usig gradient computation'''
        # number of iterations
        iterations=60

        # state sequence for randon input sequence
        self.forward()
       
       


        # gradient descent algorithm
        time1=time.time() 

        for i in range(iterations):

            cost = self.compute_traj_cost()

            # print("no of iterations completed : ", i+1 ,", current cost :",cost )

            
            F,C,c,f = self.gradients(self.x[self.N-1,0],self.x[self.N-1,1],self.x[self.N-1,2],self.x[self.N-1,3],
                                     self.u[self.N-1,0],self.u[self.N-1,1]  )

            C_x_x  = C[:4,:4].reshape(4,4)
            C_x_u = C[:4,4:].reshape(4,2)
            C_u_u = C[4:,4:].reshape(2,2)
            C_u_x  = C[4:,:4].reshape(2,4)
            c_x = c[:4].reshape(4,1)
            c_u = c[4:].reshape(2,1)


            # calculating terminal gain values
            self.K[0] = -np.linalg.inv(C_u_u) @ C_u_x
            self.k[0] = -np.linalg.inv(C_u_u) @ c_u 
  
     
            # calculating value function backwards in time

            V = C_x_x + C_x_u @ self.K[0] + np.transpose(self.K[0]) @ C_u_x + \
                np.transpose(self.K[0]) @ C_u_u @ self.K[0]
            v = c_x + C_x_u @ self.k[0]+ np.transpose(self.K[0]) @ c_u + \
                 np.transpose(self.K[0]) @ C_u_u @ self.k[0]

            # starting from state N-1
           
            for n in range(1,self.N):


                F,C,c,f = self.gradients(self.x[self.N-(n+1),0],self.x[self.N-(n+1),1],self.x[self.N-(n+1),2],self.x[self.N-(n+1),3],
                                        self.u[self.N-(n+1),0],self.u[self.N-(n+1),1])


                Q_t = C + np.transpose(F) @ V @ F
                q_t = c + np.transpose(F) @ v


                Q_x_x  = Q_t[:4,:4].reshape(4,4)
                Q_x_u = Q_t[:4,4:].reshape(4,2)
                Q_u_u = Q_t[4:,4:].reshape(2,2)
                Q_u_x  = Q_t[4:,:4].reshape(2,4)
                q_x = q_t[:4].reshape(4,1)
                q_u = q_t[4:].reshape(2,1)

                self.K[n] = -np.linalg.inv(Q_u_u) @ Q_u_x
                self.k[n] = -np.linalg.inv(Q_u_u) @ q_u


                V = Q_x_x + Q_x_u @ self.K[n] + np.transpose(self.K[n]) @ Q_u_x + \
                np.transpose(self.K[n]) @ Q_u_u @ self.K[n]


                v = q_x + Q_x_u @ self.k[n] + np.transpose(self.K[n]) @ q_u \
                + np.transpose(self.K[n]) @ Q_u_u @ self.k[n]

            self.update_traj()
            
            new_cost=self.compute_traj_cost()

            alpha=1


            ## basically using line search to eliminate numerical gradient explosions
            while new_cost-cost>100:
                alpha-=0.5
                self.update_traj(alpha)
                new_cost=self.compute_traj_cost()



        time2=time.time ()
        cost=cartpole.compute_traj_cost()
        print("computation time is :", time2-time1)
        print("optimal trajectory found for the cost function ###### ")

        print("minimum cost found :",cost)

        self.test ()


    def test(self):
        

        print("starting animation ...")


        ''' test environment for the computed state and input sequence'''

        self.testenv.reset()
        self.testenv = gym.wrappers.RecordVideo(self.testenv,'video',name_prefix="ilqr_cartpole.mp4")

        for n  in range(0,self.N):
                
                action=self.u[n]
                observation, reward, terminated, truncated, info = self.testenv.step(action[0])
        self.testenv.close()



    def plot(self):

        '''plotting function for input and state sequence'''


        with plt.style.context('seaborn-v0_8'):
            plt.figure(figsize=(8, 4))
            plt.plot(self.u[:,0],linestyle='dashdot',label="ForceOnCart")
            plt.ylabel("input N")
            plt.xlabel("time")
            plt.legend()
            plt.figure(figsize=(8, 4))
            plt.plot(self.x, linestyle='dashdot',label=('CartPosition',"CartVelocity","PoleAngle","PoleAngularVelocity"))
            plt.ylabel("states")
            plt.xlabel("time")
            plt.legend()

        plt.show()
    

if __name__ == "__main__":
    cartpole = Cartpole()
    cartpole.ilqr()
    cartpole.plot()