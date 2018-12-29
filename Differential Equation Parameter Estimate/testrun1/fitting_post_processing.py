import param_estimate as pe
import matplotlib.pyplot as plt
import numpy as np
import time
import pickle

filename = 'fitting_data'
save_data_series = [1,2,3]

for j in save_data_series:
	pkl_file = open(filename+'_'+str(j)+'.par', 'rb')
	SAVE_DATA = None
	SAVE_DATA = pickle.load(pkl_file)
	pkl_file.close()


	fig = plt.figure(j)
	ax = None
	ax2 = None
	ax = fig.add_subplot(211)

	# ------------------- EXPT DATA ---------------------------------=
	Xr, Yr = pe.load_data(plot_data=0)
	ax.scatter(Xr,Yr,3,marker="x",c="g")

	# ------------------- DATA FROM OPTIMIZATION ---------------------
	color_scheme = np.linspace(0,1,len(SAVE_DATA))
	for i in range(len(SAVE_DATA)):
		data_store = SAVE_DATA[i]
		n_set = data_store["n_set"]
		t_set = data_store["t_set"]
		p = data_store["p"]
		mse_list = data_store["mse_list"]
		print(" i = ", i)
		print("   p = ", p )
		if i==0:
			this_label = "init curve" 
		else:
			this_label = str(i)
		ax.plot(t_set,n_set, color=(1-color_scheme[i],0,color_scheme[i]), label=this_label)
		if i ==0:
			this_title = "initial G,k1,k2,k3 = " + str(p)
	
	plt.title(this_title)
	ax2 = fig.add_subplot(212)
	ax2.plot(range(len(mse_list)),mse_list)
	ax.legend()
	ax.set_xlabel("t")
	ax.set_ylabel("n")
	ax2.set_xlabel("i")
	ax2.set_ylabel("MSE")
plt.show()	

print("\nClosing Post-Processing Program...")