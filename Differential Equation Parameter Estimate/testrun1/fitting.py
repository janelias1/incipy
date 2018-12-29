import numpy as np
import kero.utils.utils as ut
import param_estimate as pe
import matplotlib.pyplot as plt
from scipy.integrate import RK45
import csv
import pickle

es = pe.DE_param_estimator()

print("----------------- Load Data -------------------------------")
Xr, Yr = pe.load_data(plot_data=0)

x0 = np.array([Yr[0]])
t_set_expt = [np.array([x]) for x in Xr]
n_set_expt = [np.array([x]) for x in Yr]
print("x0 = ",x0)
print("Xr[:5] = ",Xr[:5])
print("Yr[:5] = ",Yr[:5])
print("t_set_expt[:5] = ",t_set_expt[:5])
print("n_set_expt[:5] = ",n_set_expt[:5])

#.#

print("----------------- Prepare Model ---------------------------") 
# MODEL used to fit this data
#
# We are estimating the parameters of the following differential equations:
#   dn/dt = F(n, p) = G - k1 * n - k2 * n**2 - k3 * n**3
#   where p is the parameter (G,k1,k2,k3)
def F(y,p):
	return p[0] - p[1] * y -p[2] * y**2 - p[3] * y**3
diff_part_G = lambda x: 1
diff_part_k1 = lambda x: -x 
diff_part_k2 = lambda x: -x**2 
diff_part_k3 = lambda x: -x**3 
list_of_derivatives = [diff_part_G, diff_part_k1, diff_part_k2, diff_part_k3]
print("  Model prepared.")
#.#

do_initial_testing = 1
if do_initial_testing:
	print("---------------- Prepare Initial guess --------------------")
	# Before optimizing, try playing around the p_init values here
	# We will choose the range of uniform random values for guess p based on tests conducted here
	# Write down choices that seem good and any remarks here 
	# 
	#   1. p_init = [0,5e-2,2e-3,1.1e-7] # near the real value. all n less than expt data
	#   2. p_init = [0,5e-2,1e-3,1.1e-7] # seems to be better than 1
	#   3. p_init = [0,5e-2,0,1.1e-7] # all n above expt data 
	#   4. p_init = [0,1e-1,1e-4,1.1e-7] # small n above expt data, larger n below
	#   5. p_init = [0,1e-2,1e-3,1.1e-7] # exceedingly close! Let's vary around these values instead
	#
	#
	#

	collection_of_p_init =[
		[0,1e-3,0.5e-3,1.2e-7], # guess 1
		[0,1e-2,0,1e-7], # guess 2
		[0,0,1.5e-3,1e-7], # guess 3
	]

	fig = plt.figure()
	ax = fig.add_subplot(111)
	no_of_test = len(collection_of_p_init)
	for i in range(no_of_test):
		p_init = collection_of_p_init[i] # guess p
		time_checkpoints = t_set_expt
		def f(t,y):
			return np.array([F(y[0],p_init)])
		_,_, x_checkpoint, t_checkpoint_new = pe.RK45_wrapper(x0, time_checkpoints, f, stepsize=None)
		ax.plot(t_checkpoint_new,x_checkpoint, label="guess "+str(i+1), color=(0,1-1*(i+1)/no_of_test,1*(i+1)/no_of_test))

		# To find what is the best learning_rate
		print(" testing initial guess parameters: ",i+1)
		es.update_p(t_set_expt, n_set_expt, F, p_init, list_of_derivatives, verbose=11)
	ax.plot(Xr,Yr, color = "r", label="expt values")
	ax.legend( )
	ax.set_xlabel("t")
	ax.set_ylabel("n")

	plt.show()
#.#


# ------------------------------------------------------------------------------- #
start_op = 0
if start_op:	
	print("----------------- Start Optimizing ---------------------------")
	
	# ********* Settings *************
	p_init_max = [1e-10,0 ,0,0.9e-7] # [Gmax,k1max,k2max,k3max]
	p_init_min = [1e-10, 2e-2,2e-3,1.2e-7] # [Gmin ,k1min,k2min,k3min]
	no_of_tries = 3
	save_name = 'fitting_data'
	es.learning_rate= 1e-16
	no_of_iterations = 10
	save_interval = 1
	# ********************************

	n_set_expt_MAT = pe.RK45_output_to_list(n_set_expt) # MATRIX list form
	
	for j in range(no_of_tries):
		p_init = []
		for a,b in zip(p_init_min, p_init_max):
			p_init.append(np.random.uniform(a,b))
		p_now= p_init
		save_count = 0
		SAVE_DATA = []
		print("TRY ", j+1," / ",no_of_tries)
		mse_list = []
		for i in range(no_of_iterations + 1):
			if i>0:
				# for i =0 , initial state, do not iterate yet
				p_now = es.update_p(t_set_expt, n_set_expt, F, p_now, list_of_derivatives, verbose=0)
			
			#------------------------ FOR REALTIME OBSERVATION -----------
			def f(t,y):
				return np.array([F(y[0],p_now)])
			time_checkpoints = t_set_expt
			_,_, n_set_next, t_set_next = pe.RK45_wrapper(x0, time_checkpoints, f, stepsize=None)
			n_set_next_MAT = pe.RK45_output_to_list(n_set_next)
			mse = pe.MSE(n_set_next_MAT,n_set_expt_MAT)
			mse_list.append(mse)
			print(" i = ", i , " -- > mse = ", mse)
			# print("   p_new = ", p_now)

			#------------------------ TO BE SAVED -------------------------
			save_count = save_count + 1
			if save_count == save_interval or i==0:
				save_count = 0
				data_store = {
				"t_set":t_set_next,
				"n_set":n_set_next,
				"p":p_now,
				"learning_rate":es.learning_rate,
				"mse_list": mse_list
				}
				SAVE_DATA.append(data_store)
		output = open(save_name+"_"+ str(j+1) +".par", 'wb')
		pickle.dump(SAVE_DATA, output)
		output.close()

print("\nClosing Program...")