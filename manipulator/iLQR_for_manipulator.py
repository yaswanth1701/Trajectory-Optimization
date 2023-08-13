from manipulator_env import *
from scipy.integrate import solve_ivp
import time


class Trajectory_Optimization(Env):

    def __init__(self,initial_pos,planning_horizon,desired_coordinates,joint_vel = np.zeros(3)):

        ''' Initializing the environment and initial condtions for performing trajectory optimization
        Parameters:
        - initial_pos : Initial cartesian space coordinates for the end effector (have to convert into joint positions).
        - joint_vel : Initial joint space veclocities for the joints.
        - planning_horizon : The size for which optimization should be performed
        - desired_coordinates : The desired cartsian space coordinates of the end effector (have to convert into joint positions).
        - Solver :String type (RK45 or Euler) by default the solver for integration to get next state is 'Runge kutta' (scipy) and Euler can be also used for faster computation
        of gradients but can result in numerical errors.


        Optimization variables:

        - states : state of of the manipulator is the joint angles and joint velocities.
        - actions or input : input of the manipulator is the joint efforts or torques.
        '''
        # getting the robot parameters from the Env class
        super().__init__(initial_pos,desired_coordinates)

        print("\nUsing Symplectic Euler Integrator (default for Pybullet)\n")

        self.dt = 1/self.frequency

        print(f"Using The Itegration With Step Length : {self.dt}\n")


        # length of the planning horizon 

        self.planning_horizon = planning_horizon

        self.iterations =  10

        self.delta = 1e-5


        # initial trajectory guess for states and input (normally with zeros)

        # print(f"type of the state space variable is {type(self.states_space)}")

        self.states = np.zeros((planning_horizon,self.states_space))

        self.states[0,:] = np.hstack((self.initial_joint_pos,joint_vel))


        self.actions = np.zeros((planning_horizon,self.actions_space))

        for i in range(planning_horizon):
            self.actions[i,:] = self.initial_torque

        # specifying the desired states of end effector of the manipulator


        self.desired_joint_vel = np.zeros(self.num_rev_joints)

        self.desired_states = np.hstack((self.desired_joint_pos,self.desired_joint_vel))


        print(f"The final desired states are : {self.desired_states}")

        # creating feedback matrix and feedforward matrix for iLQR algo 

        self.K = np.zeros((planning_horizon, self.actions_space,self.states_space))
        self.k = np.zeros((planning_horizon, self.actions_space,1))


        self.Q = np.array([[500, 0 , 0 , 0 , 0 , 0],
                           [0,500 , 0 , 0 , 0 , 0],
                           [0, 0 ,500,  0 , 0 , 0],
                           [0, 0 , 0 , 100, 0 , 0],
                           [0, 0 , 0 , 0 ,100, 0],
                           [0, 0 , 0 , 0 , 0 ,100]])

        self.R = np.array([[1,0,0],
                           [0,1,0],
                           [0,0,1]])

        


        print("gettting initial trajetory states from initial guess of actions (Torques) \n")

        self.get_trajectory()

        initial_cost = self.compute_total_cost()

        print(f"Initial trajectory cost is : {initial_cost} \n")

        print("Starting trajectory optimization ...")

        self.run_trajectory_optimization()

        print("Trajectory optimization completed successfully !!!")




    def get_trajectory(self,alpha = 1):
        ''' Computes the actual states of the manipulator for the given trajectory of  action values '''

        # creating some dummy variables for computing the actual states from given actions.

        N = self.planning_horizon

        updated_states = np.zeros((self.planning_horizon,self.states_space))
        updated_states[0,:] = self.states[0,:]

        for n in range(self.planning_horizon-1):


            action = self.actions[n,:] + self.K[N-(n+1)]@(np.transpose(updated_states[n,:]-self.states[n,:])) + alpha*self.k[N-(n+1)].ravel()


            self.actions[n,:] = action.copy()


            updated_states[n+1,:] = self.get_next_state(updated_states[n,:],action)

        self.states = updated_states.copy()
       





    def get_next_state(self,current_state,action):

        '''This mehod solves for the next state of the manipulator given the current state and action
        
        
        Parameter:
        - current_state
        - action

        Return :
        - next_state : the next states of the manipulator at time step (n+1)
        '''

        qddot = self.dynamics( current_state, action)


        qdot = current_state[3:]


        next_states = current_state + self.dt * np.hstack((qdot,qddot))
        # 

        
        return next_states

    def dynamics(self,states : np.ndarray,action):

        ''' Calculation of joint acceleration and velocity from given states and torques
        
        Parameters:
        - states
        - action
        

        Purpose: this method computes the mass matrix(M(q)@qddot), gravity matrix(G(q)),coriolis and centrifugal forces (V(q,qdot)@qdot) 
        to get the current accelerations of joints.

        return:
        - gradient array : this gradient of states (position and veclocity) which are vecloctiy ad accelerations of the joints
        '''

        # seperating joint positions and velocity
        q = states[:3]
        qdot = states[3:]

        qddot_0 = np.zeros(self.num_rev_joints)

        non_linear_terms , mass_matrix =  self.get_dynamics(q , qdot,qddot_0)

        qddot = self.get_joint_acceleration(mass_matrix,non_linear_terms,action)
        


        return qddot

    def cost_function(self,state,action):
        '''
        Defining the cost function with states and actions as objective 
        
        Parameters:

        - states
        - actions

        '''


        return (state - self.desired_states).T @ self.Q @ (state - self.desired_states) + action.T @ self.R @ action


    def compute_total_cost(self):

        '''Computes the cost for entrie trajectory
        
        Return:

        - total_cost
        
        '''
        
        total_cost = 0 

        for n in range(self.planning_horizon):

            total_cost += self.cost_function(self.states[n,:],self.actions[n,:])
        
        return total_cost

    def get_Analytical_derivates(self,state,action):

        ''' Using pinocchio to the analytical derivaties of the dynamics with respect to optimization variables
        
        Returns:


        - qddot_dq : jacobian of joint acceleration with respect to joint position
        - qddot_dqdot: jacobain of joint acceleration with respect to joint velocity
        - qddot_du : jacobian of joint acceleration with respect to input'''

        q = state[:3]
        qdot = state[3:]
        torque = action

        pin.computeABADerivatives(self.model,self.data,q,qdot,torque)

        ddq_dq = self.data.ddq_dq # Derivatives of the FD w.r.t. the joint config vector
        ddq_dqdot = self.data.ddq_dv # Derivatives of the FD w.r.t. the joint velocity vector
        ddq_du = self.data.Minv # Derivatives of the FD w.r.t. the joint torque vector

        return ddq_dq , ddq_dqdot , ddq_du


    def  gradient_of_dynamics(self,initial_condtions,dim=0):

        ''' using finite difference method computes the differentiation of whole vector valued function with respect  to the  passed variable

        Parameter:

        initial_conditon : value at  which the  derivative is required (vector of state and input variables  stacked  horizontally at required  time step)


        dim  : Is  the variable with which we  want to compute the derivative (integer)


        Return :

        returns  a  row   vector (derivative value)
        
        
        '''

        

        var   =  initial_condtions.copy()
        var[dim]   =  var[dim] +  self.delta

        
 
        states  = var[:self.states_space].copy()

        input = var[self.states_space:9].copy()
    
        
        sol1 = self.get_next_state(states,input) 
    
        



        var[dim]  = var[dim]  -   2*self.delta

        states  = var[:self.states_space].copy()

        input = var[self.states_space:9].copy()
        

        sol2 = self.get_next_state(states,input) 

        




        return   (sol1  - sol2) / (2*self.delta)





    def compute_gradients(self,state,action,n):

        ''' computes the gradient of dynamics with respect to state and input variables

        Parameters:
        - State
        - Action

        Returns:
        - cost function gradient and hessian 
        - jacobain of dynamics
        '''
        cost_matrix = np.zeros((9,9))
        cost_matrix[:6,:6] = self.Q
        cost_matrix[6:,6:] = self.R
        

        # gradient vector of the cost function


        c_t = 2*cost_matrix@ np.hstack((state - self.desired_states , action)).T


        # hesssian  matrix of the cost function

        C_t = 2*cost_matrix

        F = np.zeros((self.states_space,self.actions_space+self.states_space))

    


        ddq_dq , ddq_dqdot , ddq_du = self.get_Analytical_derivates(state,action)

        # # The jacobian matrix of dynamics

        # F[:3,:3] = np.identity(self.num_rev_joints) 

        # F[:3,3:6] = self.dt * ddq_dqdot

        # F[:3,6:] = np.zeros((self.num_rev_joints,self.num_rev_joints))

        # F[3:6,:3] = self.dt * ddq_dq

        # F[3:6,3:6] = np.identity(self.num_rev_joints) + self.dt * ddq_dqdot

        # F[3:6,6:] = self.dt*ddq_du

        initial_condition = np.hstack((state,action))

        

        for i in range(self.states_space+self.actions_space):
            F[:,i] =  self.gradient_of_dynamics(initial_condition,i).T

        # print(f"the F matrix is : \n {F}")






        return c_t , C_t , F


    def line_search(self, old_cost , new_cost):

        '''To  avoid  gradient explosion problems (compares cost and finds the appropriate alpha)
        Parameters:
        -  old_cost: previous cost iteration cost 
        -  new_cost: current iteration cost 
        
        Returns:
        - nothing to return
        '''

        alpha = 0
        while new_cost >= old_cost:
    
            alpha =  alpha -  0.2
            if alpha < -1:
                break
            self.get_trajectory(alpha=alpha)
            new_cost =  self.compute_total_cost()
            print("cost in line search is :", new_cost)

        return new_cost



    def run_trajectory_optimization(self):
        
        ''' the trajectory optimization for manipulator (updates the entire action sequence)
        
        Parameters :
        - takes no parameters

        Returns :

        - Return nothing
    
        '''


        sd  =  self.states_space
        ad = self.actions_space
        N =  self.planning_horizon

       
        t1 = time.time()


        for  i  in  range(self.iterations):



            cost_old = self.compute_total_cost()
           
    
    
            c_t,C_t,F = self.compute_gradients(self.states[self.planning_horizon-1,:],
                                                      self.actions[self.planning_horizon-1,:],
                                                      self.planning_horizon-1)
    
    
            C_x_x  = C_t[:sd,:sd].reshape(sd,sd)

            C_x_u = C_t[:sd,sd:].reshape(sd,ad)
    
            C_u_u = C_t[sd:,sd:].reshape(ad,ad)

            C_u_x  = C_t[sd:,:sd].reshape(ad,sd)
    
    
            c_x = c_t[:sd].reshape(sd,1)

            c_u = c_t[sd:].reshape(ad,1)
    
    
            self.K[0] = -np.linalg.inv(C_u_u) @ C_u_x

            self.k[0] = -np.linalg.inv(C_u_u) @ c_u
          

    
    
            V = C_x_x + C_x_u @ self.K[0] + np.transpose(self.K[0]) @ C_u_x + \
                    np.transpose(self.K[0]) @ C_u_u @ self.K[0]
    
    
            v = c_x + C_x_u @ self.k[0]+ np.transpose(self.K[0]) @ c_u + \
                     np.transpose(self.K[0]) @ C_u_u @ self.k[0]

            
    
            for n in range(1,N):
    
                c_t,C_t,F   = self.compute_gradients(self.states[self.planning_horizon-(n+1)],
                                                self.actions[self.planning_horizon-(n+1)],
                                                self.planning_horizon-(n+1))
                
    
                Q_t = C_t + np.transpose(F) @ V @ F
                # print(((np.transpose(F) @ v).ravel()).shape)

                q_t = c_t + (np.transpose(F) @ v).ravel()
                

                

                Q_x_x  = Q_t[:sd,:sd].reshape(sd,sd)

                Q_x_u = Q_t[:sd,sd:].reshape(sd,ad)


                Q_u_u = Q_t[sd:,sd:].reshape(ad,ad)

                Q_u_x  = Q_t[sd:,:sd].reshape(ad,sd)


                q_x = q_t[:sd].reshape(sd,1)

                q_u = q_t[sd:].reshape(ad,1)

                self.K[n] = -np.linalg.inv(Q_u_u) @ Q_u_x

                self.k[n] = -np.linalg.inv(Q_u_u) @ q_u


                V = Q_x_x + Q_x_u @ self.K[n] + np.transpose(self.K[n]) @ Q_u_x + \
                np.transpose(self.K[n]) @ Q_u_u @ self.K[n]


                v = q_x + Q_x_u @ self.k[n] + np.transpose(self.K[n]) @ q_u \
                + np.transpose(self.K[n]) @ Q_u_u @ self.k[n]

            self.get_trajectory()
                
            new_cost = self.compute_total_cost()

            new_cost = self.line_search(cost_old,new_cost)

            

            print(f"Iteration number is : {i+1} - Trajectory cost is : {new_cost}   \n")        
        
        t2 = time.time() 
        print("total  time  taken:  ", (t2  -  t1)*1e+3,"millisecond")

        error = self.states[self.planning_horizon-20,:] - self.desired_states

        print(f"final error in states is following : {error}")


        

        print("Deploying ...")

        self.run(self.actions,TEST = True,planning_horizon= self.planning_horizon, K = self.K,optimal_states= self.states)

        print("plotting the  result \n")
        self.plot()



    def plot(self):


       '''plotting function for input and state sequence'''
       with plt.style.context('seaborn-v0_8'):
           plt.figure(figsize=(8, 4))
        #    plt.plot(self.actions,linestyle='dashdot',label=("joint_torque1","joint_torque2","joint_torque3"))
           plt.plot(self.actual_effort,linestyle='dashdot',label=("joint_torque1","joint_torque2","joint_torque3"))
           plt.ylabel("input N")
           plt.xlabel("time")
           plt.legend()
           plt.figure(figsize=(8, 4))
        #    plt.plot(self.states, linestyle='dashdot',label=("joint_pos1", "joint_pos2", "joint_pos3", "joint_vel1", "joint_vel2", "joint_3"))
           plt.plot(self.actual_states, linestyle='dashdot',label=("joint_pos1", "joint_pos2", "joint_pos3", "joint_vel1", "joint_vel2", "joint_3"))
           plt.ylabel("states")
           plt.xlabel("time")
           plt.legend()
       plt.show()


if  __name__ == "__main__":

    initial_pos = np.array([0.2,0,0.5])
    planning_horizon = 800
    desired_coordinates = np.array([-0.3,0.3,0.6])

    test = Trajectory_Optimization(initial_pos, planning_horizon, desired_coordinates)
            


        






        



        



