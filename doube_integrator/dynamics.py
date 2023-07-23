import numpy as np
import matplotlib.pyplot as plot 
class dynamic:
    def __init__(self):
        m=5
    def state_space_model(self):
        self.A=np.array([[0,1],[0,0]])
        self.B=np.array([[0],[0]])
        return (self.A,self.B)

    def step(self,x0,v0):
        state=np.array([[x0],[v0]])
        self.euler_approx()
    def euler_approx(self)