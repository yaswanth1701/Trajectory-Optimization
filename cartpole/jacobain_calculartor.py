from torch.autograd.functional import jacobian
from torch import tensor

def dynamics(x,v,theta,omega,force1 ,force2 ):
    m=0.1
    M = 1
    l = 0.5
    self.m =0.1
    self.g = 9.8
    self.l = 0.5
        self.M=1
        self.total_mass=1.1
        self.ml=self.m*self.l

    x_t=x + dt*v

    return (x1 + x2, x3*x1, x2**3)
 
#Defining input tensors
x1 = tensor(3.0)
x2 = tensor(4.0)
x3 = tensor(5.0)
 
#Printing the Jacobian
print(jacobian(f,(x1,x2,x3)))
