# We are estimating the parameters of the following differential equations:
#   dn/dt = F(n, p) = G - k1 * n - k2 * n**2 - k3 * n**3
#   where p is the parameter (G,k1,k2,k3)

import numpy as np
import kero.utils.utils as ut
from scipy.integrate import RK45
import matplotlib.pyplot as plt
import csv


class DE_param_estimator:
	def __init__(self):
		self.learning_rate = 1e-21
		return
	# l2
	def update_p(self,t_set_expt, n_set_expt, F, p_now, list_of_derivatives,
		verbose = False):
		# t_set_expt: list of numpy array
		#   for example: [array([25.2047]), array([25.279]), array([25.3533]), array([25.4277]),...]
		# n_set_expt: list of numpy array
		#   for example: [array([789.696]), array([784.109]), array([763.528]), array([726.782]),...]
		# p_now: list of parameters
		# 
		#
		# return
		#   p_next: list of parameters
 		nabla_mse = self.nabla_MSE(t_set_expt, n_set_expt, F, p_now, list_of_derivatives)
 		if verbose>10:
 			# temp = self.learning_rate*nabla_mse
 			temp2 = np.transpose(np.matrix(p_now))
 			for_recommend = []
 			if verbose>30:
	 			print("    nabla_mse = ")
	 			ut.print_numpy_matrix(nabla_mse,formatting="%s",no_of_space=10)
	 			print("    p_now =")
	 			ut.print_numpy_matrix(temp2,formatting="%s",no_of_space=10)
	 		det_list = []
	 		with np.errstate(divide='ignore'):
	 			for k in range(len(p_now)):
	 				det = 1e-2*p_now[k]/nabla_mse.item(k)
	 				if verbose>20:
		 				print("    p_now[k]/(nabla_mse[k]*100) =",det)
		 			det2 = np.log10(np.abs(det))
		 			if (det2!=np.inf) and (det2!=-np.inf) and (not np.isnan(det2)):
			 			det_list.append(det2)
		 	if verbose>20: print("      det_list = ",det_list)
		 	print("       recommended learning rate ~ ", 10**np.min(det_list))
 		p_next= np.transpose(np.matrix(p_now)) - self.learning_rate * nabla_mse
 		p_next = np.transpose(p_next).tolist()
 		p_next = p_next[0]
 		return p_next
	# L3	
	def dkdMSE(self, t_set_expt, n_set_expt, F, p_now, diff_part):
		# lambda function : d/dk d/dt of n
		# t_set_expt, n_set_expt : experimental data
		N = len(t_set_expt)
		dkmse = 0 

		def f(t,y):
			return np.array([F(y[0],p_now)])

		x0 = n_set_expt[0] # array
		time_checkpoints= t_set_expt
		x_set, t_set, x_checkpoint, t_checkpoint_new = RK45_wrapper(x0, time_checkpoints, f, stepsize=None)

		for i in range(1,len(n_set_expt)):			
			ni = x_checkpoint[i][0]
			# print(" ni =", ni)
			m = i+1
			diff_term = 0
			for j in range(m):
				diff_term = diff_term + diff_part(x_checkpoint[j][0])
			diff_term = diff_term / m 
			dkmse = dkmse + ( ni - n_set_expt[i][0])* diff_term
		dkmse = 1/N*dkmse 
		return dkmse

	def nabla_MSE(self, t_set_expt, n_set_expt, F, p_now, list_of_derivatives,
		verbose = False):
		# list_of_derivatives: list of lambda functions

		dGdMSE = self.dkdMSE(t_set_expt, n_set_expt, F, p_now, list_of_derivatives[0])
		dk1mse = self.dkdMSE(t_set_expt, n_set_expt, F, p_now, list_of_derivatives[1])
		dk2mse = self.dkdMSE(t_set_expt, n_set_expt, F, p_now, list_of_derivatives[2])
		dk3mse = self.dkdMSE(t_set_expt, n_set_expt, F, p_now, list_of_derivatives[3])
		# print("dGdMSE=", dGdMSE)
		# print("dk1mse=",dk1mse)
		# print("dk2mse=",dk2mse)
		# print("dk3mse=",dk3mse)
		nabla_mse = np.transpose(np.matrix([dGdMSE, dk1mse, dk2mse, dk3mse]))
		return nabla_mse

def load_data(plot_data=False):
	print(" --+ load_data().")

	# load data
	filename = 'sample_data'
	all_data = read_csv(filename, header_skip=0)
	all_data = [[np.float(x) for x in y] for y in all_data]

	X=[x[0] for x in all_data]
	Y=[x[1] for x in all_data]

	# customize your data loading here
	max_index = np.argmax(Y)
	relevant_data= all_data[max_index:]
	Xr = [x[0] for x in relevant_data]
	Yr = [x[1] for x in relevant_data]

	print("     data size = ", len(Xr))
	print("     peak = ", X[max_index],",", Y[max_index])
	if plot_data:
		fig = plt.figure()
		ax=fig.add_subplot(111)
		ax.scatter(X,Y,3)
		ax.scatter(X[max_index],Y[max_index])
		ax.plot(Xr,Yr,'r')
		ax.set_xlabel("t")
		ax.set_ylabel("n")
		plt.show()

	return Xr, Yr

def read_csv(filename, header_skip=1,get_the_first_N_rows = 0):
    # filename : string, name of csv file without extension
    # headerskip : nonnegative integer, skip the first few rows
    # get_the_first_N_rows : integer, the first few rows after header_skip to be read
    #   if all rows are to be read, set to zero

    # example
    # xx = read_csv('tryy', 3)
    # xx = [[np.float(x) for x in y] for y in xx] # nice here
    # print(xx)
    # print(np.sum(xx[0]))
    
    out = []
    with open(str(filename)+'.csv') as csv_file:
        data_reader = csv.reader(csv_file)
        count = 0
        i = 0
        for row in data_reader:
            if count < header_skip:
                count = count + 1
            else:
                out.append(row)
                if get_the_first_N_rows>0:
                	i = i + 1
                	if i == get_the_first_N_rows:
                		break
    return out

def MSE(Y_set,Y0_set):
	# Y_set is a list of Y
	# Y0_set is a list of Y0
	# Assume Y_set and Y0_set have the same size, otherwise error will be raised
	#
	# Y (numpy matrix) is a vector: computed data from model
	# Y0 (numpy matrix) is a vector: observed/true data
	# Assume Y and Y0 have the same size, otherwise error will be raised
	# Assume all entries are real
	
	mse = 0
	n = len(Y_set)
	if not (n==len(Y0_set)):
		print("MSE(). Error. Inconsistent dimension between Y_set and Y_set0. Return nothing.")
		return	
	for i in range(n):
		Y = Y_set[i]
		Y0 = Y0_set[i]
		Nout1 = len(Y.tolist())
		Nout2 = len(Y0.tolist())
		if not (Nout1==Nout2):
			print("MSE(). Error. Inconsistent dimension between Y and Y0. Return nothing.")
			return
		# now compute the norms between true and model output values
		for j in range(Nout1):
			mse = mse + (np.abs(Y0.item(j)-Y.item(j)))**2
	mse = mse / (2*n)

	return mse

def RK45_wrapper(x0, time_checkpoints, f, stepsize=None):
	# example
	#
	#
	#
	# def f(t,y):
	# 	# dx/dt = f(x,t)
	# 	return np.array([-1*y[0]])
	# x0 = np.array([1.0]) # initial value
	# time_checkpoints = np.linspace(0,10,11)
	# x_set, t_set, x_checkpoint, t_checkpoint_new = RK45_wrapper(x0, time_checkpoints, f, stepsize=None)

	# t_checkpt_diff = []
	# for t,t_new in zip(time_checkpoints, t_checkpoint_new):
	# 	t_checkpt_diff.append(abs(t-t_new))
	# print("max checkpt diff = ", np.max(t_checkpt_diff))

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

def RK45_output_to_list(xx):
	return [np.matrix((x.tolist())[0]) for x in xx]