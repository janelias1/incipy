import numpy as np
import matplotlib.pyplot as plt
import scipyode_header as sh
import pickle
from scipy.optimize import least_squares

# TOGGLES:
initial_guess_mode = 0
optimize_mode = 1
# SETTINGS:
save_name = "20190110_cubic"

# --------------- PART 1 ----------------
#
# Load data, prepare the differential equation

def f(t,n, f_args): # function for pendulum motion with friction 
	dndt = [f_args[0] - f_args[1] * n - f_args[2] * n**2 - f_args[3] * n**3]
	return dndt
print("\n------------- LOADING DATA ----------------\n")
Xr, Yr = sh.load_data(plot_data=0)

x0 = np.array([Yr[0]])
t_set_expt = [np.array([x]) for x in Xr]
n_set_expt = [np.array([x]) for x in Yr]
print("x0 = ",x0)
print("Xr[:5] = ",Xr[:5])
print("Yr[:5] = ",Yr[:5])
print("t_set_expt[:5] = ",t_set_expt[:5])
print("n_set_expt[:5] = ",n_set_expt[:5])

# --------------- PART 2 ----------------
#
# try out initial guesses

# initial guess [G, k1, k2, k3]
collection_of_initial_guesses = [ 
 [0,1e-3,0.5e-3,1.2e-7],
 [0,1e-2,0,1e-7],
 [0,0,1.5e-3,1e-7]
]

if initial_guess_mode:
	print("\n------------- INITIAL TESTING ----------------\n")
	color_scheme = np.linspace(0,1,len(collection_of_initial_guesses))
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(Xr,Yr,3,color='g',label="expt data")
	for i in range(len(collection_of_initial_guesses)):
		f_args0 = collection_of_initial_guesses[i]
		# print(f_args0)
		this_label = "initial guess" + str(i+1)
		_, _, x_test, t_test = sh.RK45_wrapper(x0, t_set_expt, f, f_args0, stepsize=None)		
		ax.scatter(t_test,x_test,3,color=(1-color_scheme[i],0,color_scheme[i]),label=this_label)
	ax.legend()
	ax.set_xlabel("t")
	ax.set_ylabel("n")	

if optimize_mode:
	def target_function(f_args):
		# "checkpoint" here refers to the intended data points
		#   since RK45 will generate more points in between the checkpoints.
		#   What happens is just that with smaller time steps in between the checkpoints,
		#   we can compute the checkpoints more precisely.
		_, _, x_checkpoint, t_checkpoint_new = sh.RK45_wrapper(x0, t_set_expt, f, f_args, stepsize=None)
		
		collection_of_f_i = []
		# each i is a data point in this particular example
		for x1, x2 in zip(x_checkpoint,n_set_expt):
			this_norm = np.linalg.norm(np.array(x1)-np.array(x2))
			collection_of_f_i.append(this_norm)
		return np.array(collection_of_f_i)
	print("\n------------- OPTIMIZATION STAGE ----------------\n")

	for i in range(len(collection_of_initial_guesses)):
		f_args0 = collection_of_initial_guesses[i]
		print(" initial guess = ", f_args0)
		optimized_res = least_squares(target_function,f_args0)
		f_args_optimized = optimized_res.x

		SAVE_DATA = {
			"initial_guess": f_args0,
			"optimized": f_args_optimized
		}

		print(" + optimized result = ",f_args_optimized)
		output = open(save_name+"_"+ str(i+1) +".par", 'wb')
		pickle.dump(SAVE_DATA, output)
		output.close()
	print("\n------------- OPTIMIZATION STAGE ----------------\n")


plt.show()