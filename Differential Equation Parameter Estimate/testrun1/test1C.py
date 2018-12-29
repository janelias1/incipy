import numpy as np
from scipy.integrate import RK45
import matplotlib.pyplot as plt

def f(t,y):
	# dx/dt = f(x,t)
	return np.array([-1*y[0]])

x0 = np.array([1.0]) # initial value
# t0 = 0  # initial time
# t1 = 10 # final time

def RK45_wrapper(x0, time_checkpoints, f, stepsize=None):

	if stepsize is None:
		stepsize = 0.01*(time_checkpoints[-1]-time_checkpoints[0])
	t_set = [time_checkpoints[0]]
	x_set = [x0]
	t_checkpoint_new = [time_checkpoints[0]]
	x_checkpoint = [x0]

	for i in range(len(time_checkpoints)-1):	
		t1 = time_checkpoints[i+1]
		if i>0:
			x0 = ox.y
			t0 = ox.t
			
		else:
			t0 = time_checkpoints[i]
		ox = RK45(f,t0,x0,t1,max_step=stepsize)
		
		while ox.t < t1:
			msg = ox.step()
			t_set.append(ox.t)
			x_set.append(ox.y)
		x_checkpoint.append(ox.y)
		t_checkpoint_new.append(ox.t)
	return x_set, t_set, x_checkpoint, t_checkpoint_new

# all data by DE (differential equation)
time_checkpoints = np.linspace(0,10,11)
x_set, t_set, x_checkpoint, t_checkpoint_new = RK45_wrapper(x0, time_checkpoints, f, stepsize=None)

t_checkpt_diff = []
for t,t_new in zip(time_checkpoints, t_checkpoint_new):
	t_checkpt_diff.append(abs(t-t_new))
print("max checkpt diff = ", np.max(t_checkpt_diff))

# all data, analytic
t0 = time_checkpoints[0]
t1 = time_checkpoints[-1]
t_set2 = np.linspace(t0,t1,100)
f_int = lambda t: np.exp(-t)
x_set2 = [f_int(t_set2[i]) for i in range(len(t_set2))]

fig = plt.figure()
ax=fig.add_subplot(111)
ax.scatter(t_set,x_set,facecolors='none',edgecolors='r',label="all data by DE")
ax.scatter(t_checkpoint_new,x_checkpoint,20,'g',label="check points")
ax.plot(t_set2,x_set2,label="all data, analytic")
ax.set_xlabel("t")
ax.set_ylabel("x")
ax.legend()
plt.show()