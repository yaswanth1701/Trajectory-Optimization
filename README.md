# Trajectory-Optimization
This repo contains trajectory-optimisation  on some basic systems (ex: pendulum,cartpole ,quadrotors and manipulators). Will  be implementing algorithms like finite horizon LQR, iLQR , AL-ilQR,Box DDP and DIRCOL.
### Algorithms  :
- [X] Infinite horizon LQR:



###### with external disturbance(0-50N)  -
<p align="center">

<img src="https://github.com/yaswanth1701/Trajectory-Optimization/assets/92177410/0f8d7961-70f0-45c5-9640-71cb30740881.gif" width="300" height="200">
</p>








- [X] Iterative LQR :
<p align="center">

<img src="https://github.com/yaswanth1701/Trajectory-Optimization/assets/92177410/3be5d0fe-c398-4398-9109-6f9766d525f9" width="200" height="200">
</p>

- [X] Iterative LQR (with line search) :

      
<p align="center">

<img src="https://github.com/yaswanth1701/Trajectory-Optimization/assets/92177410/5b540218-bf16-4102-86a6-97186a7f60da.gif" width="300" height="200">
</p>

- [X] Iterative LQR (with finite-horizon LQR for trajectory tracking) :
<p align="center">
<img src="https://github.com/yaswanth1701/Trajectory-Optimization/assets/92177410/61f5d702-7077-464f-b5cc-54334013f191.gif" width="600" height="300">
</p>

- [X] MPC using DIRCOL :

<p align="center">
<img src="https://github.com/yaswanth1701/Trajectory-Optimization-DDP-iLQR/assets/92177410/8bb4e595-98e5-479a-8be5-65e9feabe37f" width="600" height="300">
</p>

for code refer [here](https://github.com/yaswanth1701/MPC-for-Mobile-Robot).

### Current environments:
- [x] Cartpole (gym)
- [x] Pendulum (gym)
- [x] Quadrotor (PyBullet)
- [x] Turtlebot3 (gazebo)
- [ ] 5 link Biped (matplotlib)
### References:
- [Notes](https://www.notion.so/Trajectory-Optimisation-DDP-iLQR-7b680055afff496ba324bc03827f32e3?pvs=4)
- [Advanced robotics by Pieter Abbeel (iLQR)](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa19/)
- [Optimal control (CMU AL-iLQR)](https://youtu.be/qGoGGSpg9Fs)
- [Underactuated robotics (For non linear dynamics)](https://youtube.com/playlist?list=PLkx8KyIQkMfVVMjf9FtTojfUvKNqscEN9)
- [Control-Limited Differential Dynamic Programming (Box DDP)](https://homes.cs.washington.edu/~todorov/papers/TassaICRA14.pdf)
- [really good explanation for DDP](http://www.imgeorgiev.com/2023-02-01-ddp/#:~:text=It%20is%20an%20extension%20of,non%2Dlinear%20trajectory%20optimisation%20problems.)
