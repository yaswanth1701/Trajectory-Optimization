from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np 
from matplotlib import cm
# x1=np.arange(0,10,0.1)
# x2=np.arange(0,10,0.1)
# u=np.arange(0,10,0.1)
x = np.linspace(0, 10, 10)
y = np.linspace(0, 10, 10)
z = np.linspace(0, 10, 10)
Q1,Q2,R=(1,1,1)
mesh = np.meshgrid(x, y, z)
# print(len(mesh[0]))
nodes = list(zip(*(dim.flat for dim in mesh)))
cost=np.zeros(len(nodes))
for i,node in enumerate(nodes):
    cost[i]=node[0]*node[0]*Q1+node[1]*node[1]*Q2+node[2]*node[2]*R
fig = plt.figure()
ax = fig.gca(projection='3d')
my_col = cm.jet(cost)
surf = ax.plot_surface(nodes[:,0],nodes[:,1], nodes[:,2], rstride=1, cstride=1, facecolors = my_col,
        linewidth=0, antialiased=False)