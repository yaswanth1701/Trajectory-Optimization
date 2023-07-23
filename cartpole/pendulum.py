import matplotlib.pyplot as plt
import numpy as np
import gym
from torch.autograd.functional import jacobian
from torch import tensor


class Cartpole:
    def __init__(self):
        '''all the  neccessary parameters  for the system and  
        lqr control and cost  parameters'''

        # system parameters
        self.m =1
        self.g = 9.8
        self.l = 1
 
        # planning horizon
        self.N = 50

        # loading environment
        self.env = gym.make('Pendulum-v1', g=9.81)
        self.testenv = gym.make('Pendulum-v1', g=9.81, render_mode="rgb_array")
        # time step size
        self.dt = 0.05
        # random input sequence
        self.u = np.ones((self.N, 1))

        # lqr gains
        self.K = np.zeros((self.N, 1,2))
        self.k = np.zeros((self.N, 1,1))

        # random state sequence
        self.x = np.zeros((self.N, 2))
        
        #  updated states  sequence
        self.updatedx= np.zeros((self.N, 2))
        self.x[0, :] = [np.pi, 0]
        self.updatedx[0,:]=[np.pi,0]

        # dynamics 
        self.f = np.zeros((4, 1))
        # cost  function weights

        
        # cofficient for each system variable
        self.Q =np.array([[20,0],[0,1]])
        self.R=np.array([[1]])
    


    def forward(self):
            '''updating states and  torques'''
            self.env.reset()
            for n  in range(0,self.N):
                self.updatedx[n,:]=np.array([np.arctan2(np.sin(self.env.state[0]) , np.cos(self.env.state[0]) )
                                             ,self.env.state[1]])

                action=self.u[n]

                observation, reward, terminated, truncated, info = self.env.step(action)
            self.x=self.updatedx.copy()

            self.env.close()
    def update_traj(self):
        self.env.reset()
        for n in range(0,self.N):
            self.updatedx[n,:]= np.array([np.arctan2(np.sin(self.env.state[0]) , np.cos(self.env.state[0]) )
                                             ,self.env.state[1]]).ravel()

            state=np.array([np.arctan2(np.sin(self.env.state[0]) , np.cos(self.env.state[0]))
                                             ,self.env.state[1] ]).ravel()

            action = self.u[n]+self.K[self.N-(n+1)]@(np.transpose(state-self.x[n]))+self.k[self.N-(n+1)]

            self.u[n]=action.copy()

            observation, reward, terminated, truncated, info = self.env.step(action)

        self.x=self.updatedx.copy()



    def statespacemodel(self):
        '''| next state |                      dynamics                      |
           |------------|----------------------------------------------------|
           | theta      | theta+dt*omega                                     | 
           | omega      | omega + dt*(3 / 2)*((g*sin(theta)/ l)+(3*u/(m *l^2)))| 
        '''
        pass

    def compute_traj_cost(self):
        # cost associated with state variables
        Q=self.Q
        R=self.R
        trajectory_cost=0
        for n in range(self.N):
            trajectory_cost+=self.x[n,:] @ Q @ np.transpose(self.x[n,:]) + self.u[n,:] @ R @ np.transpose(self.u[n,:])
        return trajectory_cost



    def gradients(self,theta,omega,torque):
        '''' calculating gradients of system dynamics and cost function
        dyngrad: is jacobian of the system dynamics
        self.c : is the first derivative of the cost function
        self.C : is the second derivative of the cost function or jacobian of the self.c
        '''
        m=self.m
        l=self.l
        g=self.g
        dt=self.dt 
        ## jacobian of dynamics
        dyngrad  =   np.array([ [ 1 ,                          dt  ,              0 ],
                                [ dt*3*g*np.cos(theta)/(2*l) ,  1  ,  dt*(3/m*l**2) ]  ])
        f = np.zeros((2,1))
       
        #gradient of cost function

        self.c = 2*np.array([[self.Q[0,0] * theta], [self.Q[1,1] * omega],[self.R[0,0]*torque]])

        # jocabian of cost function

        self.C= 2*np.array([[self.Q[0,0],0,0],[0,self.Q[1,1],0],[0,0,self.R[0,0]]])
        return dyngrad,self.C,self.c,f



    def ilqr(self):
       
        iterations=20
        self.forward()
      
       


        # gradient descent algorithm

        for i in range(iterations):
            cost= self.compute_traj_cost()
            print("no of iterations completed : ", i ,"current cost :",cost )
            
            F,C,c,f = self.gradients(self.x[self.N-1,0],self.x[self.N-1,1],self.u[self.N-1,0])

            C_x_x  = C[:2,:2].reshape(2,2)
            C_x_u = C[:2,2].reshape(2,1)
            C_u_u = C[2,2].reshape(1,1)
            C_u_x  = C[2,:2].reshape(1,2)
            c_x = c[:2].reshape(2,1)
            c_u = c[2].reshape(1,1)


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


                F,C,c,f = self.gradients(self.x[self.N-(n+1),0],self.x[self.N-(n+1),1],self.u[self.N-(n+1),0])


                Q_t = C + np.transpose(F) @ V @ F
                q_t = c + np.transpose(F) @ v


                Q_x_x  = Q_t[:2,:2].reshape(2,2)
                Q_x_u = Q_t[:2,2].reshape(2,1)
                Q_u_u = Q_t[2,2].reshape(1,1)
                Q_u_x  = Q_t[2,:2].reshape(1,2)
                q_x = q_t[:2].reshape(2,1)
                q_u = q_t[2].reshape(1,1)

                self.K[n] = -np.linalg.inv(Q_u_u) @ Q_u_x
                self.k[n] = -np.linalg.inv(Q_u_u) @ q_u


                V = Q_x_x + Q_x_u @ self.K[n] + np.transpose(self.K[n]) @ Q_u_x + \
                np.transpose(self.K[n]) @ Q_u_u @ self.K[n]


                v = q_x + Q_x_u @ self.k[n] + np.transpose(self.K[n]) @ q_u \
                + np.transpose(self.K[n]) @ Q_u_u @ self.k[n]

            self.update_traj()
        cost=cartpole.compute_traj_cost()
        print("optimal trajectory found for the cost function ###### ")
        print("minimum cost found :",cost)
        self.test ()

    def test(self):
        self.testenv = gym.wrappers.RecordVideo(self.testenv,'video',name_prefix="ilqr_cartpole.mp4")
        self.testenv.reset()
        print("starting animation ...")
        for n  in range(0,self.N):
                
                action=self.u[n] 
                observation, reward, terminated, truncated, info = self.testenv.step(action)
        self.testenv.close()



    def plot(self):
        with plt.style.context('seaborn-v0_8'):
            plt.figure(figsize=(8, 4))
            plt.plot(self.u)
            plt.ylabel("input N")
            plt.xlabel("time")
            plt.figure(figsize=(8, 4))
            plt.plot(self.x, linestyle='dashdot',label=("pole angle","pole angular velocity"))
            plt.ylabel("states")
            plt.xlabel("time")
            plt.legend()

        plt.show()
    
        

   


if __name__ == "__main__":
    cartpole = Cartpole()
    cartpole.ilqr()
    
    cartpole.plot()