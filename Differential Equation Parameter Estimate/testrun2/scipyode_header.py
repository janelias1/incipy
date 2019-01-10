import numpy as np
from scipy.integrate import RK45
import matplotlib.pyplot as plt
import csv

def RK45_wrapper(x0, time_checkpoints, f, f_args, stepsize=None):
	if stepsize is None:
		stepsize = 0.01*(time_checkpoints[-1]-time_checkpoints[0])
	t_set = [time_checkpoints[0]]
	x_set = [x0]
	t_checkpoint_new = [time_checkpoints[0]]
	x_checkpoint = [x0]

	f_temp = lambda t,y : f(t,y, f_args)

	for i in range(len(time_checkpoints)-1):	
		t1 = time_checkpoints[i+1]
		if i>0:
			x0 = ox.y
			t0 = ox.t
			
		else:
			t0 = time_checkpoints[i]
		ox = RK45(f_temp,t0,x0,t1,max_step=stepsize)
		
		while ox.t < t1:
			msg = ox.step()
			t_set.append(ox.t)
			x_set.append(ox.y)
		x_checkpoint.append(ox.y)
		t_checkpoint_new.append(ox.t)
	return x_set, t_set, x_checkpoint, t_checkpoint_new

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