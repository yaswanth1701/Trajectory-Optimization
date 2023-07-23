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
        self.env=gym.make('CartPole-v1',render_mode="rgb_array")
        self.state_space_parameters()
        S=linalg.solve_continuous_are(self.A, self.B, self.Q,self.R)
        self.K=np.linalg.inv(self.R)@np.transpose(self.B)@S
    def state_space_parameters(self):
        self.A=np.array([[0,1,0,0],[0,0,-self.m*self.g/self.M,0],[0,0,0,1],[0,0,(self.M+self.m)*self.g/(self.l*self.M),0]])
        self.B=np.array([[0],[1/self.M],[0],[-1/(self.M*self.l)]])
        self.Q=np.array([[10,0,0,0],[0,1,0,0],[0,0,10,0],[0,0,0,1]])
        self.R=np.array([[0.001]])
    def lqr(self,state):
        u=- self.K@state
        return float(u[0])
    def forward_pass(self):
        self.env.reset()
        state=np.transpose(np.array(self.env.state))
        # terminated=False
        truncated=False
        i=0
        self.states=np.zeros((500,4))
        self.u=np.zeros(500)
        self.env = gym.wrappers.RecordVideo(self.env,'video',name_prefix="ilqr_cartpole.gif")
        while not (truncated):
            action=self.lqr(state)+np.random.randint(0,50)
            # action=0
            observation, reward, terminated, truncated, info=self.env.step(action)
            state=np.transpose(np.array(observation))
            self.states[i,:]=state
            print(state[2])
            self.u[i]=action
            i+=1
    def plotting(self):
        with plt.style.context('seaborn-v0_8'):
            plt.figure(figsize =(8, 4))
            plt.plot(self.u)
            plt.ylabel("input N")
            plt.xlabel("time")
            plt.figure(figsize =(8,4))
            plt.plot(self.states,label=('CartPosition',"CartVelocity","PoleAngle","PoleAngularVelocity"))
            plt.ylabel("states")
            plt .xlabel("time")
            plt.legend()
        plt.show()


        
if __name__ == "__main__":
    cartpole=Cartpole()
    cartpole.forward_pass()
    cartpole.plotting()
