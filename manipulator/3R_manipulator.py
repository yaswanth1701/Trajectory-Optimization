import pybullet as p
import pybullet_data
import time
import math
import numpy as np

physicsClient = p.connect(p.DIRECT)#or p.DIRECT for non-graphical version

# set visualization
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

# load world

p.setAdditionalSearchPath(pybullet_data.getDataPath()) #used by loadURDF (plane)
planeId = p.loadURDF("plane.urdf")
p.setGravity(0,0,-10)

# load robot model

manipulatorPos = [0,0,0]
StartOrientation = p.getQuaternionFromEuler([0,0,0])
manipulator = p.loadURDF("/home/yaswanth/optimal_control/manipulator/urdf/rrr_arm.urdf ",manipulatorPos, StartOrientation)


p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

number_of_joints = p.getNumJoints(manipulator)
print(f"number of joints is {number_of_joints}")

joint_info = p.getJointInfo(manipulator,2)

print(f"type of the joint and its info is :{joint_info}")

joint_states = p.getJointStates(manipulator,np.array([0,1,2]))

print(f"the joint states of the each joint is or the joint angle {joint_states}")

number_of_links_plane = p.getLinkState(planeId,0)

print(f"info of the plane {number_of_links_plane}")


# fixing the base to plane or ground
constraint_no =  p.createConstraint(planeId,-1,manipulator,-1,p.JOINT_FIXED,[0,0,1],[0,0,0],[0,0,0])
print(f"here is the constraint info of the base plane and base link of the manipulator: {p.getConstraintInfo(constraint_no)} ")
## reseting the joint states of the manipulator
# p.resetJointState(manipulator,0,np.pi/4)
# p.resetJointState(manipulator,1,-np.pi/2)
# p.resetJointState(manipulator,2,-np.pi/2)
# p.resetJointStatesMultiDof(manipulator,[0,1,2],[np.pi/4,-np.pi/4,-np.pi/4])
print(f"resetting the joint states of the manipulator, new state of joint is {p.getJointState(manipulator,0)}")
angles = np.linspace(0,-np.pi/4,240)

coordinate = np.array([1,0,1])
joints_state = p.calculateInverseKinematics(manipulator,2,coordinate)
print(f"joint angles for reaching this coordinate is {joints_state}")

M = np.array(p.calculateMassMatrix(manipulator,joints_state))

print(f"The mass matrix is following : {M}")
print(f"size of the mass matrix is : {M.shape}")
i=0
while 1:
    p.stepSimulation()
    Pos, Orn = p.getBasePositionAndOrientation(manipulator)


    p.setJointMotorControlArray(manipulator,[0,1,2],p.POSITION_CONTROL,joints_state)
    # i = i+1
    # if i >=240:
    #     break
    time.sleep(1/240)

    
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