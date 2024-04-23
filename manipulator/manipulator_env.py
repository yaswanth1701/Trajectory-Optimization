import numpy as np 
import matplotlib.pyplot as plt 
import math as m 
import pybullet as p
import pybullet_data
import pinocchio as pin
from sys import argv
from os.path import dirname, join, abspath
import time 
from robot_descriptions.loaders.pybullet  import load_robot_description


class Env():

    def __init__(self,initial_position,desired_position):

        ''' envrionment setup and initialization 
        
        Parameters:
        - initial_position : initial position of the end effector of the manipulator in cartesian coordinates (x,y,z)
        '''

        self.frequency = 240


        print(f" \n Initial cartesian Coordinates Of The End Effector are {initial_position} \n")
        # interface setup of the environment
        physicsClient = p.connect(p.GUI)

        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)


        self.model_path = "franka_panda/panda.urdf"


        # for loading the plane in environment 
        p.setAdditionalSearchPath(pybullet_data.getDataPath()) #used by loadURDF (plane)

        planeId = p.loadURDF("plane.urdf")

        # set garvity 

        self.gravity =  -10

        p.setGravity(0,0,self.gravity)
    

        # settting up base pose and orientation of the manipulator

        manipulatorPos = [0,0,0]

        StartOrientation = p.getQuaternionFromEuler([0,0,0])

        self.manipulator = p.loadURDF(self.model_path,useFixedBase=1)


        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 1)


        p.addUserDebugLine(initial_position,desired_position,lineWidth = 3)


        # getting number of joints in the manipulator

        self.update_joint_info()


        # getting initial joint space positions
        self.initial_joint_pos = np.array(p.calculateInverseKinematics(self.manipulator,11 ,initial_position,maxNumIterations = 50))
        self.desired_joint_pos = np.array(p.calculateInverseKinematics(self.manipulator,11 ,desired_position,maxNumIterations = 50))
        
        

        print(f"\nInitial Joint Positions Or Angles Are : {self.initial_joint_pos} \n")

        print(f"\n Desired joint space position are : {self.desired_joint_pos} \n")


        #  setting up initial joint position for the manipulator
    
        for i,n in enumerate(self.rev_joints_index):

            p.resetJointState(self.manipulator,n,self.initial_joint_pos[i])
        
        
        # states for the manipulator (DOF) and action also 
         
        self.states_space = 18

        self.actions_space = 9

        # getting intial joint positions and velocities
        joint_states = p.getJointStates(self.manipulator,self.rev_joints_index)


        #print(f"\nThe Dimension Of The State Space Is : {self.states_space} And The Dimension Of The Action Space Is : {self.actions_space} \n")

        self.initial_vel = np.zeros(9)

        self.initial_acc = np.zeros(9)


        print("Environment Is Ready !!!!! \n")

        print("Enabling torque control ...")

        self.enable_torque_control()

        print("setting up pinocchio ")
        self.setup_pinocchio()

        print("\nGetting initial gravity compensation torques")

        self.initial_torque, mass_matrix = self.get_dynamics(self.initial_joint_pos,self.initial_vel,self.initial_acc)

        #self.initial_acc = self.get_joint_acceleration(mass_matrix,self.initial_torque,self.initial_torque)
        #initial_acc,self.initial_torque, mass_matrix = self.forward_dynamics(self.initial_joint_pos,self.initial_vel,self.initial_acc)

        print("\nInitial configuration gravity compensation vector : \n {initial_torque} \n".format(initial_torque = self.initial_torque))

        print("\nInitial configuration mass matrix is : \n {mass_matrix}  \n".format(mass_matrix=mass_matrix.shape))

        print("\nInitial joint accelerations are following : \n {initial_acc}  \n".format(initial_acc = self.initial_acc))


    

        print("Starting the simulation ...\n")


    def setup_pinocchio(self):

       

        pinocchio_model_dir = dirname(dirname(str(abspath(__file__))))
        urdf_filename = pinocchio_model_dir+'/manipulator/urdf/rrr_arm.urdf ' if len(argv)<2 else argv[1]
        
        self.model    = pin.buildModelFromUrdf(urdf_filename)
        
        print(f"{self.model.name} loaded successfully in pinocchio")

        self.data = self.model.createData()

        print("pinocchio setup completed model loaded successfully !!!")

       
       
   
    def forward_dynamics(self,joint_positions,joint_velocities,joint_torques):
        ''' Using pinocchio to get mass matrix and joint acceleration for given position and velocity of the
        manipulator (forward dynamics). Basically this helps to compute analytical derivatives of dynamics so,
         preferred over pybullet's default functions.


        Parameters:
        - joint position
        - joint velocity
        - joint torque
        
        Returns:
        
        - non-linear effects compensating torques 
        - joint accelerations
        - mass matrix for given joint position
        '''

         # calculating mass matrix
        q = joint_positions
        M = pin.crba(self.model, self.data, q)
        

        # calculating Coriolis, Centrifugal, Gravity torque vectors
        qdot = joint_velocities
        qddot_zero = np.zeros(self.model.nv)
        non_linear_effects = pin.rnea(self.model, self.data, q, qdot, qddot_zero)

        qddot = np.linalg.inv(M)@(joint_torques - non_linear_effects)


        return qddot  , non_linear_effects , M



    def update_joint_info(self):

        self.number_of_joints =  p.getNumJoints(self.manipulator)

        print(f"Number of joints are : {self.number_of_joints}")

        for n in range(self.number_of_joints):
            joint_info = p.getJointInfo(self.manipulator,n)
            print(f"\nJoint index is :{joint_info[0]} and type is :{joint_info[2]}")


        self.rev_joints_index = [n for n in range(self.number_of_joints) if p.getJointInfo(self.manipulator,n)[2] == 0]
        self.rev_joints_index.append(9)
        self.rev_joints_index.append(10)

        self.num_rev_joints = len(self.rev_joints_index)
        

        print(f"\nIndex of revolute joints in the urdf {self.rev_joints_index} and number of revolute joints in urdf are {self.num_rev_joints}")




    def enable_torque_control(self):
        "Enabling torques control in pybullet which is not set by default"

        p.setJointMotorControlArray(self.manipulator,self.rev_joints_index, p.VELOCITY_CONTROL, forces=np.zeros(self.num_rev_joints))



    def get_dynamics(self, joint_pos : np.ndarray, joint_vel : np.ndarray , acceleration : np.ndarray):
        
         
        '''To compute initial joint positions gravity compensation vector and mass matrix'''

        # coverting to numpy array to lists for inverse dynamics

        pos_list = joint_pos.tolist()
        
        vel_list = joint_vel.tolist()

        acc_list = acceleration.tolist()


        # if qddot is passed as zero then this returns the Coriolis, Centrifugal, Gravity torque terms.

        torque = np.array(p.calculateInverseDynamics(self.manipulator,pos_list,vel_list,acc_list))
        

        # This is to get the joint space mass matrix 

        mass_matrix = np.array(p.calculateMassMatrix(self.manipulator,pos_list))

        return  torque , mass_matrix

    def get_joint_acceleration(self,mass_matrix,non_terms,applied_torques):

        acc = np.linalg.inv(mass_matrix)@(applied_torques - non_terms)

        return acc





    def step_env(self,command ,control_mode = p.POSITION_CONTROL):


        ''' This function executes exactly one action
        
        Parameters :
        - command : a array of joint position or joint angles or joint efforts/torques
        - control_mode : specify the control mode (POSITION_CONTROL or VELOCITY_CONTROL or TORQUE_CONTROL)
        '''

    


        p.setJointMotorControlArray(self.manipulator,self.rev_joints_index,control_mode,forces = command)
        p.stepSimulation()
        time.sleep(4/self.frequency)


    def run(self,joint_command,TEST = False , planning_horizon = 1, K = np.zeros(1),optimal_states = np.zeros(1)):

        ''' This function should be called to execute the entire desired trajectory.

        By default it hold on to the initial specified carstein space coordinates
        
        Parameters:

        - joint_command : array of joint efforts/torques for entire planning horizon (size should be planning_horizon x dimension of action space))
        - TEST : boolean value , if testing a trajectory this should be assigned as True !!
        - planning horizon : number of time step for which the trajectory is planned (total_time = planning_horizon / self.frequency)
        - K : feedback gain matrix of LQR control to precise  track trajectories
        - Optimal_states : To calculate the feedback torque required as damping and stiffness are not accounted in formulation of the dynamics.
        '''
        print(f"Executing the desired commands and testing is {TEST} !!!!  \n")


        N = planning_horizon

        self.actual_effort = np.zeros((planning_horizon,self.actions_space))

        self.actual_states = np.zeros((planning_horizon,self.states_space))



        if not TEST:
            
            
            command = self.initial_torque

            while planning_horizon:
                states = p.getJointStates(self.manipulator,self.rev_joints_index)
                print(f"joint states are {states}")
                self.step_env(command,p.TORQUE_CONTROL)

        
        else :

            print(f"planning horizon size is {planning_horizon} and expected time to execute trajectory is {planning_horizon/self.frequency} seconds")
        
            
            for n in range(planning_horizon):
                

                optimal_state = optimal_states[n]

                states = p.getJointStates(self.manipulator,self.rev_joints_index)
                current_pos = []
                current_vel = []
                for state in states:
                    current_pos.append(state[0])
                    current_vel.append(state[1])


                current_state = np.array([current_pos, current_vel]).ravel()

            

                command = joint_command[n] + self.K[N-(n+1)]@(np.transpose(current_state-optimal_states[n,:]))

                self.actual_effort[n]  = command
                self.actual_states[n]  =  current_state
            

                self.step_env(command,control_mode= p.TORQUE_CONTROL)

            print("Terminating the exection of optimal trajectory ...")

        print("Finished The Execution !!!")


if __name__ == "__main__":
    
    initial_pos = np.array([-0.7,0,0.5])


    test = Env(initial_pos,initial_pos)

    test.run(joint_command = test.initial_joint_pos)




            
        
        








        

















