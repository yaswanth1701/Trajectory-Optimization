import numpy as np 
phases =np.pi/12
print(np.arctan2(np.sin(phases), np.cos(phases)))
def next_state(self,state,input):
      x=state[0]+state[1]*self.dt
      v=state[1]+(input[0]+self.m*np.sin(state[2])*(self.l*np.square(state[3])-self.g*np.cos(state[2])))*self.dt/(self.M+self.m*np.square(np.sin(state[2])))
      theta=state[2]+state[3]*self.dt
      omega=state[1]+self.dt*(-input[0]*np.cos(state[2])-self.m*self.l*np.square(state[2])*np.sin(state[2])*np.cos(state[2])+(self.M+self.m)*self.g*np.sin(state[2]))/((self.M+self.m*np.square(np.sin(state[2])))*self.l)
      next_state=np.array([[x],[v],[theta],[omega]])
      return np.transpose(next_state)