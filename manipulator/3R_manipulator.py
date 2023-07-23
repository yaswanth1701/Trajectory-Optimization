import pybullet as p
import pybullet_data
import time
import math
import numpy as np

physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version

# set visualization
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

# load world

p.setAdditionalSearchPath(pybullet_data.getDataPath()) #used by loadURDF (plane)
planeId = p.loadURDF("plane.urdf")
p.setGravity(0,0,-10)

# load robot model

YunaStartPos = [0,0,0.5]
YunaStartOrientation = p.getQuaternionFromEuler([0,0,0])
Yuna = p.loadURDF("/home/yaswanth/optimal_control/manipulator/urdf/rrr_arm.urdf ",YunaStartPos, YunaStartOrientation)

p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
# p.saveWorld("yuna_quickworld")

while 1:
    p.stepSimulation()
    YunaPos, YunaOrn = p.getBasePositionAndOrientation(Yuna)

    
p.disconnect()

"""
# get motors list
joint_num = p.getNumJoints(Yuna)
print("joint_number:",joint_num)
actuators = []
for j in range(joint_num):
    info = p.getJointInfo(Yuna,j)
    if info[2] == p.JOINT_REVOLUTE:
        actuators.append(j)
        print(info[1])
print(actuators)   # [12, 13, 14, 17, 18, 19, 22, 23, 24, 27, 28, 29, 32, 33, 34, 37, 38, 39]   in  1,3,5,2,4,6 order
"""