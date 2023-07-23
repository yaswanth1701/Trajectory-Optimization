import gym
from scipy import linalg
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
class Cartpole:
    def __init__(self):
        self.m=0.1
        self.g=9.8
        self.l=0.5
        self.M=1
        self.N=150#planning horizon
        self.env=gym.make('CartPole-v1',render_mode="rgb_array")
        self.J=np.zeros(self.N) # cost per each time step
        self.K=np.zeros((self.N,1,4)) # k for each time step 
        self.P=np.zeros((self.N,4,4)) # P for each time step
        self.U=np.zeros(self.N)
        self.dt=0.02
        self.terminal_state=np.array([[0],[0],[0],[0]])
        self.state_space_parameters()
        self.backwardpass()
    def state_space_parameters(self):
        self.A=np.array([[1,self.dt,0,0],[0,1,self.dt*(-self.m*self.g/self.M),0],[0,0,1,self.dt],[0,0,self.dt*(self.M+self.m)*self.g/(self.l*self.M),1]])
        self.B=np.array([[0],[self.dt*1/self.M],[0],[self.dt*-1/(self.M*self.l)]])
        self.Q=np.array([[10,0,0,0],[0,1,0,0],[0,0,10,0],[0,0,0,1]])
        self.R=np.array([[1]])
    def backwardpass(self):
        self.terminal_cost=np.transpose(self.terminal_state)@self.Q@self.terminal_state
        self.P[0]=self.Q
        for n in range(1,self.N):
            self.K[n]=-np.linalg.inv(self.R+np.transpose(self.B)@self.P[n-1]@self.B)@np.transpose(self.B)@self.P[n-1]@self.A
            self.P[n]= self.Q+np.transpose(self.K[n])@self.R@self.K[n]+np.transpose(self.A+self.B@self.K[n])@self.P[n-1]@(self.A+self.B@self.K[n])
    def lqr(self,state,n):
            self.U[n]=self.K[self.N-n-1]@state
            self.J[self.N-n-1]=state@self.P[self.N-n-1]@np.transpose(state)
            print(self.J[self.N-n-1])
            return float(self.U[n])

    def forward_pass(self):
        self.env = gym.wrappers.RecordVideo(self.env,'video',name_prefix="finite_lqr_cartpole")
        self.env.reset()
        state=np.transpose(np.array(self.env.state))
        # terminated=False
        truncated=False
        t=0
        self.states=np.zeros((self.N,4))
        self.u=np.zeros(self.N)
       
        for n in range(self.N):
            action=self.lqr(state,n)
            observation, reward, terminated, truncated, info=self.env.step(action)
            state=np.transpose(np.array(observation))
            self.states[t,:]=state
            t+=1
        print(self.terminal_cost)
    def plotting(self):
        with plt.style.context('seaborn-v0_8'):
            plt.figure(figsize =(8, 4))
            plt.plot(self.J[:-1],linestyle="--")
            plt.plot(self.U,linestyle="--")
            plt.ylabel("input N")
            plt.xlabel("time")
            plt.figure(figsize =(8,4))
            plt.plot(self.states,linestyle="--",label=('CartPosition',"CartVelocity","PoleAngle","PoleAngularVelocity"))
            plt.ylabel("states")
            plt .xlabel("time")
        plt.show()


        
if __name__ == "__main__":
    cartpole=Cartpole()
    cartpole.forward_pass()
    cartpole.plotting()
