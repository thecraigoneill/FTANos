
from FTANos import *
import numpy as np
import matplotlib.pyplot as plt

# The following script runs an inversion of a digitised dispersion curve.
# The example follows a Nelder Mead inversion in O'Neill et al. 2024, which is reproducible.
# Following that inversion, we invoke a MCMC inversion, which is statistical, and thus varies each time.
# The latter is used to constrain a posteri velocity distributions.



# First we input the filename of seismic data file, and the dispersion file (here called 'figname').
# The latter will be used as a prefix for figures later. 

sfile = "example_trace.sac"

figname = sfile +".disp"

# Create a velocity array in the standard CPS/Disba format (thickness (km), Vp (km/s), Vs (km/s) and rho (g/cm3))
# This will be used/varied for initial conditions in the Nelder Mead inversion approach.

velMod =  np.array(  [
					[2.0e-3, 0.80, 1.00, 1.00],
					[2.0e-3, 0.80, 1.00, 1.00],
					[2.0e-3, 0.80, 1.00, 1.00],
					[2.0e-3, 0.80, 1.00, 1.00],
					[2.0e-3, 0.80, 1.00, 1.00],
					[2.5e-3, 0.80, 1.00, 1.00],
					[2.5e-3, 1.80, 1.00, 1.50],
					[2.5e-3, 1.80, 1.00, 1.50],
					[2.5e-3, 1.80, 1.50, 1.50],
					[5.0e-3, 1.80, 1.50, 1.50],
					[5.0e-3, 1.80, 1.50, 1.50],
					[5.0e-3, 1.80, 1.50, 1.50],
					[5.0e-3, 2.80, 1.50, 1.50],
					[5.0e-3, 2.80, 1.50, 1.50],
					[5.0e-3, 2.80, 1.50, 1.50],
					[5.0e-3, 2.80, 1.50, 1.50],
					])

# Create repeated depth array for plotting velocity models
d0 = np.array([])
d1 = np.cumsum(velMod[:,0])
d2 = d1-np.copy(velMod[:,0])

for i in range(len(d1)):
    d0 = np.append(d0,d2[i])
    d0 = np.append(d0,d1[i])

d0 *=1e3
############################################################

###Initiate plots

plt.figure(figsize=(12,6),dpi=300)
plt.rcParams.update({'font.size': 10})

# Run FTANos inversion using Nelder-Mead simplex method
# We define a maximum velocity here (1.5km/s) beyond which models are penalised.
# This first command initialises the model. We supply the Velocity Model velMod, 
# and maximum velocity penalty term max_vs,as well as the dispersion file name
out_inv = FTAN_Invert(fig_file=figname,velMod=velMod,max_vs=1.5)

# Now we run the inversion. The Nelder Mead instantiation runs 18 initial starting conditions by default.
# This can be changed for more robust runs, see descriptions in main code. 
velModa, Vs_a, std_a, best, disp, disp_obs = out_inv.Amoeba_crawl()
print(np.size(Vs_a),np.size(std_a))

#Plot results
plt.subplot(121)
V_plot = np.repeat(Vs_a,2)
plt.fill_betweenx(d0,V_plot-np.repeat(std_a,2),V_plot+np.repeat(std_a,2),alpha=0.5,color="xkcd:mint green")
plt.plot(V_plot,d0,label=str(np.min(best)),alpha=1.0,linewidth=2.0,color="xkcd:black")
plt.legend(fontsize=7)
plt.ylim(np.max(d0),0)
plt.xlim(0,3.5)


plt.subplot(122)
plt.plot(disp[0],disp[1],linewidth=2.0,color="xkcd:black",label="Inversion")
plt.plot(disp_obs[0],disp_obs[1],linewidth=2.0,color="xkcd:black",linestyle="--",label="Data")
plt.legend()


# Save outputs
c = np.zeros( (len(std_a),1) ) #Here we need to stack the standard deviation onto the veloity model for saving.
c[:,0] = std_a
np.savetxt("best_veloMod2.dat",np.hstack((velModa,c)))
np.savetxt("best_disp_curves2.dat",np.c_[disp_obs[0],disp_obs[1],disp[0],disp[1]])

plt.savefig("test_amoeba_suite2.png")
plt.show()

# Optionally we can also run a Markov-Chain Monte Carlo inversion. This can be performed initially, or as
# a second step after the initial Nelder-Mead simplex inversion has converged on a sensible model. 
# The advantage of the latter is that a sensible step size can be applied to map out the a posteri distributions.
# Using this as an initial inversion requires large step sizes to cover the parameter space, 
# which may not match the a priori distribution of uncertainty. Use with nuance. 
# Uncomment the following lines to run the MCMC inversion

out_inv = FTAN_Invert(fig_file=figname,velMod=velModa,n_burn=1,n_ite=2000,step_size=0.1)
vel_mod, L1s = out_inv.go_for_a_walk()

# Note the velocity model is for an ensemble
# The output velocity model can be imaged similarly to the previous example. 
# The velocity model is stored in this step in the file "MCMC_results.dat"
# The format is d1 (depth, mean_Vs (ie. mean of ensemble),stdv (standard deviation of ensemble),
# lowest_Vs (ie. best fit)
# If the ensemble model can be forward modelled, the resulting dispersion curve is saved in the file 
# "mcmc_inversion_result.dat"
# The optimal dispersion curve will likewise be archived in a file called:
# "mcmc_optimal_model_dispersion.dat"

