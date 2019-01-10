import numpy as np
import matplotlib.pyplot as plt
import pickle
import scipyode_header as sh

# SETTINGS
load_name = "20190110_cubic"
serial_number = [1,2,3] 



print("\n------------- LOADING DATA ----------------\n")
def f(t,n, f_args): # function for pendulum motion with friction 
	dndt = [f_args[0] - f_args[1] * n - f_args[2] * n**2 - f_args[3] * n**3]
	return dndt
Xr, Yr = sh.load_data(plot_data=0)
x0 = np.array([Yr[0]])
t_set_expt = [np.array([x]) for x in Xr]
n_set_expt = [np.array([x]) for x in Yr]
print("x0 = ",x0)
print("Xr[:5] = ",Xr[:5])
print("Yr[:5] = ",Yr[:5])
print("t_set_expt[:5] = ",t_set_expt[:5])
print("n_set_expt[:5] = ",n_set_expt[:5])



print("\n------------- PLOTTING ----------------\n")
color_scheme = np.linspace(0,1,len(serial_number))

for j in range(len(serial_number)):
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(Xr,Yr,3,color='g',label="expt data")
	i = serial_number[j]
	this_label = "initial guess" + str(i+1)
	pkl_file = open(load_name+'_'+str(i)+'.par', 'rb')
	SAVE_DATA = None
	SAVE_DATA = pickle.load(pkl_file)
	pkl_file.close()
	# print(SAVE_DATA)
	f_args_initial = SAVE_DATA["initial_guess"]
	f_args_optimized = SAVE_DATA["optimized"]
	_, _, x_op, t_op = sh.RK45_wrapper(x0, t_set_expt, f, f_args_optimized, stepsize=None)	
	print("Serial number:",j+1, " - Optimized parameters = " )
	print(" ",f_args_optimized)
	title_msg = load_name +":"+ str(j+1)
	ax.plot(t_op,x_op,color=(1-color_scheme[j],0,color_scheme[j]),label=this_label)
	ax.legend()
	ax.set_title(title_msg)
	ax.set_xlabel("t")
	ax.set_ylabel("n")

plt.show()