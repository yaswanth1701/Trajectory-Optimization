import numpy as np
import pinocchio as pin
# from robot_descriptions.loaders.pinocchio import load_robot_descripti
from sys import argv
from os.path import dirname, join, abspath
import time 

pinocchio_model_dir = dirname(dirname(str(abspath(__file__))))
urdf_filename = pinocchio_model_dir+'/manipulator/urdf/rrr_arm.urdf ' if len(argv)<2 else argv[1]

model    = pin.buildModelFromUrdf(urdf_filename)

print('model name: ' + model.name)
data = model.createData()
print(data)


# print(f"link information of the  manipulator {data.joints}")
# manipulator = load_robot_description("edo_description")
# model = manipulator.model
# data = manipulator.data

# # Inertia matrix with CRBA
q = np.array([ 0.05066588, -1.2378624, -2.37199467] )
print(q)
M = pin.crba(model, data, q)

print(M)

# # Bias torques with RNEA
print(model.nv)
v = np.zeros(model.nv)
zero_accel = np.zeros(model.nv)
tau_0 = pin.rnea(model, data, q, v, zero_accel)
print(tau_0)
# # Forward dynamics by solving the linear system
tau = 42  -  tau_0
print(f"tau for the joints  is {tau}")
a = np.linalg.solve(M, tau - tau_0)
print(a)

t1 = time.time()
pin.computeABADerivatives(model,data,q,v,tau)

ddq_dq = data.ddq_dq # Derivatives of the FD w.r.t. the joint config vector
ddq_dv = data.ddq_dv # Derivatives of the FD w.r.t. the joint velocity vector
ddq_dtau = data.Minv
t2 = time.time()

print(f"time taken for gradient calculation is : {t2-t1} seconds")

print(f"position derivative \n{ddq_dq} \n velocity derivative \n {ddq_dv}  \n  torque derivative \n {ddq_dtau}")


