# FTANos core code Nov 2023. 
# Developed by Craig O'Neill and Ao Chang, QUT. Acknowledge support of the GSQ.
# Distributed under MIT licence accompanying distribution.
# Accompanies paper "Utilising frequency-time analysis (FTAN) of surface waves for geotechnical and dam investigations"


import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.fft import rfft, irfft, fft, ifft, fftfreq
from numba import jit
from obspy import read
from obspy.io.sac.sactrace import SACTrace
from scipy.optimize import minimize
from scipy.interpolate import interp1d, griddata
from scipy.signal import find_peaks
from disba import PhaseDispersion, GroupDispersion


class FTANos(object):
	def __init__(self,
				 fre1 = 10,
				 fre2 = 100,
				 vel1 = 100,
				 vel2 = 1400,
				 dist = 30,
				 alpha = 25,
				 dt = 0.00025,
				 filename = None
					):

		self.fre1 = fre1
		self.fre2 = fre2
		self.vel1 = vel1
		self.vel2 = vel2
		self.alpha = alpha     
		self.dt = dt
		self.dist = dist
		self.filename = filename
		self.st = read(self.filename, format="SAC")
		self.tr = self.st[0].detrend()
		self.x = self.tr.data
		self.dom = 1/(len(self.x)*self.dt)
		# else: 
		# 	raise ValueError('The file format can not be recognized!')
		

	def times(self,):
		t = np.arange(1e-12, np.size(self.x)*self.dt, self.dt)
		return t

	def periods(self,):
		self.T1 = 1.0/self.fre2
		self.T2 = 1.0/self.fre1
		p = np.linspace(self.T1, self.T2, 40)
		return p

	def FTAN_a(self,):
        # FTAN routine. Some parts cannibalised from CPS/AFTAN under BSD licence. Largely rewritten. 
        # Some python inspired by other FTAN distros but again largely rewritten. 
        # Inherits structure "self" containing time-series data x, and sample rate dt, periods.
        # The FTAN also needs the band filter width alpha, which is predefined (or use the default)
        # Ammplitude map scaled as per CPS and FTAN. 
        # Returns an array of amplitude for plotting.
		amplitude = np.zeros(shape=(len(self.periods()), len(self.x)))
		#apply Fourier transformation
		xi = fft(self.x)
		# array of frequencies
		freq = fftfreq(len(xi), d=self.dt)
		freq_n = np.arange(0, len(self.x), 1)*self.dom
		# filter signal if needed
		# xi[freq < 0] = 0.0
		# xi[freq > 0] *= 2.0

		for iperiod, T0 in enumerate(self.periods()):
			f0 = 1.0/T0
			xi_f0 = xi*np.exp(-self.alpha*((freq-f0)/f0)**2)
			#apply Fourier transformation back to time domain
			xi_f1 = ifft(xi_f0)/len(self.x)
			xi_f2 = np.copy(xi_f0)
			#filling amplitude and phase of column
			amplitude[iperiod, :] = 60.0* np.log10(np.abs(xi_f1)/(np.max(np.abs(xi_f1))-np.min(np.abs(xi_f0))))
			amax = -1.0e10
			for amp in amplitude[iperiod, :]:
				if amp > amax:
					amax = amp
			amplitude[iperiod, :] = amplitude[iperiod, :] + 100.0 - amax
			i = 0
			for amp in amplitude[iperiod, :]:
				if amp < 40.0:
					amplitude[iperiod, i] = 40.0
				i += 1
			if iperiod == (len(self.periods())-1):
				print(np.c_[freq, amplitude[iperiod, :]])

			#phase[iperiod, :] = np.angle(xi_f1)
		return amplitude

	def plot_FTAN(self,):
        # Function to create an FTAN plot
        # Regrids data onto a finer grid for presentation
        # Inherits structure "self", which has periods and v range, as well as amplitude from FTAN routine.
        # Creates a png FTAN map named " self.filename+".png" ", with filename being inherited from input.
		amp = self.FTAN_a()
		v = self.dist/self.times()
		X, Y = np.meshgrid(self.periods(), self.times())
		V = self.dist/Y
		X_new, Y_new = np.meshgrid(self.periods(), np.linspace(self.vel1, self.vel2, 4000))
		A_new = griddata((X.ravel(), V.ravel()), amp.T.ravel(), (X_new, Y_new), method='nearest')
		np.set_printoptions(threshold=sys.maxsize)

		ax1 = plt.subplot2grid((1,4), (0,0), colspan=3)
		im1 = ax1.imshow(A_new, origin='lower', extent=[self.T1,self.T2,self.vel1,self.vel2], aspect='auto')
		ax1.set_xlabel("Period (s)")
		ax1.set_ylabel("Group velocity (m/s)")   		
		plt.colorbar(im1, orientation="horizontal")

		ax2 = plt.subplot2grid((1,4),(0,3))
		ax2.plot(self.x, self.times(), color="xkcd:deep purple", linewidth=2)
		ax2.set(xticklabels=[])
		ax2.set_ylim(np.min(self.times()), np.max(self.times()))
		plt.tight_layout()
		fig_file = self.filename+".png"
		plt.savefig(fig_file)


	def plot_digitise(self,):
        # Function to create an FTAN plot. Inherits the structure "self".
        # Uses known range of periods and frequencies to scale mouse clicks on FTAN plot.
        # Creates a scaled dispersion curve, and saves it under self.filename+".disp"
		print("Click points starting from [0,0] [0,1] [1,0]:") 
		print("First click the origin, maximum of x axis, and maximum of y-axis, in that order.")
		print("All clicks after the first three clicks are treated as data.")
		print("Hit escape to exit")
		# ####the user clicks over 3 times (must include [0,0], [0,1], [1,0]) on the figure and return the coordinates of each click in a list
		x = plt.ginput(n=-1, timeout=0, show_clicks=True, mouse_stop="escape")
		plt.close()
		print("Coordinates of the clicked points", np.array(x))
		x = np.asarray(x)
		xminP = x[0,0]
		yminP = x[0,1]
		xmaxP = x[1,0]
		ymaxP = x[2,1]
		points_x = x[3:,0]
		points_y = x[3:,1]
		# #########normalize the coordinates in the scale of periods and velocity##########
		X = ((points_x - xminP)/(xmaxP - xminP))*(self.T2 - self.T1) + self.T1
		Y = ((points_y - yminP)/(ymaxP - yminP))*(self.vel2 - self.vel1) + self.vel1

		# disp_file = "{0:s}_{1:s}.disp".format(self.tr.stats.station, self.tr.stats.channel)
		disp_file = self.filename+".disp"
		np.savetxt(disp_file, np.c_[X,Y])

class MCMC:
	def __init__(self,
				 alpha  = 0.5,		#Recommended 0.5
				 beta   = 1.02,		#Recommended 1.1		 
				 n_burn = 2000,
				 n_ite  = 5000,
				 fig_file = None,
				 step_size  = 0.25,	#Recommended 0.25 (there is a multiplier above)
				 step_floor = 0.08,  #Recommended 0.15 (150m/s)
				 velMod = np.array( [
									[2.0e-3, 0.80, 2.00, 1.00],
									[2.0e-3, 0.80, 2.00, 1.00],
									[2.0e-3, 0.80, 2.00, 1.00],
									[2.0e-3, 0.80, 2.00, 1.00],
									[2.0e-3, 0.80, 2.00, 1.00],
									[2.5e-3, 0.80, 2.00, 1.00],
									[2.5e-3, 1.80, 2.00, 1.50],
									[2.5e-3, 1.80, 2.00, 1.50],
									[2.5e-3, 1.80, 2.50, 1.50],
									[2.5e-3, 1.80, 2.50, 1.50],
									[2.5e-3, 1.80, 2.50, 1.50],
									[2.5e-3, 1.80, 2.50, 1.50],
									[5.0e-3, 2.80, 2.50, 1.50],
									[5.0e-3, 2.80, 2.50, 1.50],
									[5.0e-3, 2.80, 2.50, 1.50],
									[5.0e-3, 2.80, 2.50, 1.50],
									]),

				 velModa = np.array([
									[ 0.005 , 0.6440 , 0.3554 , 1.9406 ],
									[ 0.005 , 0.6502 , 0.3901 , 1.9729 ],
									[ 0.005 , 0.6854 , 0.4897 , 2.0520 ],
									[ 0.005 , 0.9118 , 0.7430 , 2.1968 ],
									[ 0.005 , 1.0188 , 0.9056 , 2.2656 ],
									[ 0.005 , 1.5862 , 1.0310 , 2.3106 ],
									[ 0.005 , 1.6743 , 1.1720 , 2.3551 ],
									[ 0.005 , 1.7908 , 1.2537 , 2.3786 ],
									[ 0.005 , 1.9585 , 1.3710 , 2.4096 ],
									[ 0.000 , 1.9585 , 1.3710 , 2.4096 ],
									])
				 ):

		self.alpha  = alpha
		self.beta   = beta		
		self.n_burn = n_burn
		self.n_ite  = n_ite
		self.velMod   = velMod
		self.vel_orig = velModa
		self.fig_file   = fig_file
		self.step_size  = step_size
		self.step_floor = step_floor
		self.t = np.genfromtxt(self.fig_file, usecols=0)
		self.disp_curve = np.genfromtxt(self.fig_file, usecols=1)/1e3

	#@jit(nopython=True)
	def L1_norm(self, vel):
		try:
			pd = GroupDispersion(*vel.T)
			cpr = pd(self.t, mode=0, wave="rayleigh")
			vo = cpr[1]
			to = cpr[0]
			fm = interp1d(to, vo, fill_value="extrapolate")
			v2 = fm(self.t)
			# print("...L1 worked")
		except:
			v2 = np.ones_like(self.t)*1e9
			to = self.t
			# print("...L1 bombed")
		grad = (np.sum(np.abs(np.gradient(v2)))/np.size(v2))
		L1 = np.sum(np.abs(self.disp_curve-v2))
		return L1

	#@jit(nopython=True)
	# def noise_deadener(self, Vs0a, noise):
	# 	for i in range (len(Vs0a)):
	# 		if i==0:
	# 			local_mean = (Vs0a[i] + Vs0a[i+1])/2.0
	# 		elif i==(np.size(Vs0a)-1):
	# 			local_mean = (Vs0a[i-1] + Vs0a[i])/2.0
	# 		else:
	# 			loca_mean = (Vs0a[i-1] + Vs0a[i] + Vs0a[i+1])/3.0

	# 		if Vs0a[i] > (local_mean + 3*np.std(Vs0a)):
	# 			noise[i] = -1*np.abs(noise[i])
	# 		elif Vs0a[i] < (local_mean + 3*np.std(Vs0a)):
	# 			noise[i] = 1*np.abs(noise[i])
	# 		return noise

	#@jit(nopython=True)
	def moving_average(self, a, n=3):
		self.n = n
		ret = np.cumsum(a, dtype=float)
		ret[self.n:] = ret[self.n:] - ret[:-self.n]
		return ret[self.n -1:]/self.n

	#@jit(nopython=True)
    # This is a Markov-Chain Monte Carlo approach for inverting the dispersion curve.
    # Routine inherits structure "self" including dispersion curve (self.disp_curve from FTAN), and initial velocity model (self.velMod)
    # Forward model uses disba libraries, which are a jit python distro based on the CPS forward modelling approach. 
    # Note we use an empirical scaling of Vp and rho from Vs to reduce the problem to Vs fit.
    # Produces a npy binary file of all post-burn MCMC instances.
    # Creates a file of dispersion results in ensemble median, stdv, and optimal fit solution
    # Also creates a png plot of the MCMC Vs profile results, as well as dispersion fit and L1 evolution. 
    # The disba routine can be sensitive to irregular Vs structures and may crash. Sometimes ensemble median solution also has no forward model solution and inversion fails.
    # If this happens, try running again (MCMC will vary each time, to a degree), or if it continues, you might need to vary the inversion parameters defined at the start of the problem. 
	def go_for_a_walk(self,):
		vel_v = self.velMod
		velMod1 = np.copy(vel_v)
		d0 = np.array([])
		d1 = np.cumsum(vel_v[:,0])
		d2 = d1-np.copy(vel_v[:,0])
		
		L1_old  = self.L1_norm(vel_v)
		for i in range(len(d1)):
			d0 = np.append(d0,d2[i])
			d0 = np.append(d0,d1[i])

		d0a = np.array([])		
		d1a = np.cumsum(self.vel_orig[:,0])
		d2a = d1a - np.copy(self.vel_orig[:,0])
		for i in range(len(d1a)):
			d0a = np.append(d0a,d2a[i])
			d0a = np.append(d0a,d1a[i])

		# this burns in and gets velMod close for statistics run
		step_size_1 = self.step_size
		Vs0 = np.copy(vel_v[:,2])
		for i in range(self.n_burn):
			L1 = self.L1_norm(vel_v)
			print("%sth Burn-in iteration :"%i, "L1-norm:", round(L1,1), "VS", np.round(np.sum(self.velMod[:,2])))
			if  i==0:
				L1_old = L1
			elif L1 < L1_old:
				#Accept
				L1_old = L1
				Vs0 =  np.copy(vel_v[:,2])
			elif L1 >= L1_old:
				rdm = np.random.uniform(0.0, 1.0, 1)[0]
				# Want to randomly reward if only a little bigger
				# This dice term can blow out, but < should catch silly increases
				# If L1 close to L1_old, 2nd term should be small, and dice ~1
				dice = np.abs((1-(np.abs(L1-L1_old)/L1_old)))
				if dice < rdm:
					#Accept anyway
					L1_old = L1
					Vs0 = np.copy(vel_v[:,2])
					#print(" .  in here///",rdm,dice,np.max(Vs0))
				if L1_old > 40:
					step_size_1 *= 1.01
					if self.step_size > 8.0 :
						step_size_1 = 8.0
				elif L1_old < 8:
					step_size_1 *= 0.99
					if step_size_1 < 0.1:
						step_size_1 = 0.1
				# Now move 
				Vs0a = Vs0
				stdv = self.step_floor + self.alpha*Vs0a**self.beta
				#add the artificial noise
				local_mean = self.moving_average(Vs0a)
				means = -0.5 * (Vs0a[1:-1] - local_mean)/local_mean
				#Note these are pertubatiosn to local mean!
				means2 = np.hstack(([0], means, [0]))
				noise = np.random.normal(means2, stdv, np.size(Vs0a))*step_size_1
				#Works out local mean, makes sure noise is within +/- 3 stdv
				#noise = noise_deadener(Vs0a,noise) 
				vel_v[:,2] = np.abs(Vs0a + noise)
				# Garnders relationships etc
				Vp = (Vs0 + 1.164)/0.902   
				rho = (310 * (Vp * 1000)**0.25)/1000
				vel_v[:,1] = np.abs(Vp)
				vel_v[:,3] = np.abs(rho)

		step_size_1 = self.step_size
		Vs0=np.copy(vel_v[:,2])

		plt.figure(figsize=(13,4))
		plt.subplot(131)
		L1array = np.array([])
		L1_lowest = 10.e3
		for i in range(self.n_ite):
			L1 = self.L1_norm(vel_v)
			print(i," of ", self.n_ite," L1=", round(L1,2), "old:", round(L1_old,2), "VS:", round(np.sum(self.velMod[:,2])))
			if  i==0:
			    L1_old = L1
			    vel_olds = vel_v[:,2]
			elif L1 < L1_old:
			    #Accept
				L1_old = L1
				L1array = np.append(L1array,L1)
				vel_olds = np.vstack((vel_olds,vel_v[:,2]))
				Vs0 = np.copy(vel_v[:,2])
				Vs02 = np.repeat(Vs0,2)
				plt.plot(Vs02, d0, color="xkcd:beige", alpha=0.1)
				if (L1 < L1_lowest):
						L1_lowest = L1
						vel_lowest = np.copy(vel_v)
			    #print(i," of ",n," L1=",L1, "old:",L1_old)
			elif L1 >= L1_old:
				rdm = np.random.uniform(0.0,1.0,1)[0]
				dice =  np.abs((1 - ( np.abs(L1 - L1_old)/L1_old)))
			    #print(" ... random things:",rdm,L1_old/L1) #This is letting way too many poor things past
				if  dice < rdm:
			        #Accept anyway
					L1_old = L1
					L1array = np.append(L1array,L1)
					if (np.any(~np.isnan(vel_v[:,2]))):
						vel_olds=np.vstack((vel_olds,vel_v[:,2]))
						Vs0 = np.copy(vel_v[:,2])
						Vs02 = np.repeat(Vs0,2)
						#print("\t Inhere:",i," of ",n," L1=",L1, "old:",L1_old)
					plt.plot(Vs02, d0*1e3, color="xkcd:beige", alpha=0.1)
			# Now move
			if  L1_old > 40:
				step_size_1 *= 1.01
				if  step_size_1 > 8.0:
					step_size_1 = 8.0
			elif L1_old < 8:
				step_size_1 *= 0.99
				if  step_size_1 < 0.1:
					step_size_1 = 0.1

			Vs0a=Vs0
			# Want to vary mean of distro, to account for variations from local mean.
			stdv = self.step_floor + self.alpha*Vs0a**self.beta
			local_mean = self.moving_average(Vs0a)
			means = -0.5*(Vs0a[1:-1] - local_mean)/local_mean
			means2 = np.hstack(([0],means,[0]))
			noise = np.random.normal(means2,stdv,np.size(Vs0)) * step_size_1 
			#noise = noise_deadener(Vs0a,noise)

			# Make sure no NANs
			tmp =  np.abs(Vs0a+noise)
			if (np.sum(tmp) > 0.0):  #test for nan
				#print("JOY",tmp)
				vel_v[:,2] = tmp
			    # Garnders relationships etc
				Vp = (Vs0a + 1.164)/0.902
				rho=(310*(Vp*1000)**0.25)/1000
				vel_v[:,1]=np.abs(Vp)
				vel_v[:,3]=np.abs(rho)
	    #L1array = np.append(L1array,L1)
		np.save("vel_set.npy", vel_olds)
		print("Vels_old", np.shape(vel_olds[::-1]))
		print("Any dodge Vs?", np.any(np.isnan(vel_olds)))
		# Calculate MEDIAN PROFILE
		nn=round(self.n_ite/2)
		# MEAN/MEDIAN VS
		v100 = vel_olds[nn:]
		mean_Vs = np.median(vel_olds[100:], axis=0)
		print("MEANS:",mean_Vs, np.shape(v100))
		stdv = np.std(vel_olds[100:],axis=0)
		last_Vs = vel_olds[-1,:]
		lowest_Vs = vel_lowest[:,2]
		print("Last Line",last_Vs)
		#For plotting
		meanVs02 = np.repeat(mean_Vs, 2)
		lowest_Vs2 = np.repeat(lowest_Vs, 2) #vel_olds[-1],axis=0]
		np.savetxt(str(self.fig_file)+"MCMC_results.dat",np.c_[d1,mean_Vs,stdv,lowest_Vs])
		stdv_plus  = meanVs02 + np.repeat(stdv,2)
		stdv_minus = meanVs02 - np.repeat(stdv,2)
		d0m = d0*1e3
		plt.plot(lowest_Vs2,  d0m, color="xkcd:purple",    alpha=1.0, linewidth=3, label="Optimal fit")
		plt.plot(meanVs02,  d0m, color="xkcd:black", alpha=1.0, linewidth=3, label="MCMC Ensemble Inversion")
		plt.plot(stdv_plus, d0m, color="xkcd:black",     alpha=1.0, linewidth=2, linestyle="--")
		plt.plot(stdv_minus,d0m, color="xkcd:black",     alpha=1.0, linewidth=2, linestyle="--")

		median_Vs = np.mean(vel_olds,axis=0)
		medianVs02 = np.repeat(median_Vs,2)
		#plt.plot(medianVs02,d0,color="xkcd:blood red",alpha=1.0,linewidth=3,label="MCMC Inversion (Median)")

		vs_orig  = self.vel_orig[:,2]
		vs_origa = velMod1[:,2]		
		v_orig_2 = np.repeat(vs_orig,2)
		v_orig_2a = np.repeat(vs_origa,2)
		print("Vels_old", vel_olds)
		plt.plot(v_orig_2a, d0m,    color="xkcd:green", alpha=1.0, linewidth=1.5, linestyle="--", label="Starting model")
		#plt.plot(v_orig_2,  d0a*1e3,color="xkcd:salmon",alpha=1.0, linewidth=3, label="CPS Inversion")
		plt.legend(framealpha=0.15)
		plt.xlabel("V (km/s)")
		plt.ylabel("Depth (m)")
		plt.xlim(0,5)
		#plt.xlim(0,np.max(Vs02 + 0.1))
		plt.ylim(np.max(d1*1e3),0)

		plt.subplot(132)
		# Trying for MEAN VS
		vel_v[:,2]=mean_Vs 
		Vp = (Vs0 + 1.164)/0.902
		rho=(310*(Vp*1000)**0.25)/100
		vel_v[:,1]=np.abs(Vp)
		vel_v[:,3]=np.abs(rho)

		plt.plot(self.t, self.disp_curve, color="xkcd:lightgreen", alpha=1.0, linewidth=2, linestyle="--", label="Data")
		try:
			pd = GroupDispersion(*vel_v.T)
			cpr = pd(self.t, mode=0, wave="rayleigh") #Make sure to evaluate at same t you have disp curve measured at
			vo=cpr[1]
			to=cpr[0]
			np.savetxt("mcmc_inversion_result.dat",np.c_[to,vo])
			plt.plot(to, vo, color="xkcd:black", alpha=1.0, linewidth=3, label="MCMC Ensemble Inversion")
		except:
			print("The dispersion curve did not converge!! Try varying the step number, or beta.")
			print("Ze problem was ere:",vel_v[:,2])


		vel_v[:,2]=vel_lowest[:,2] 
		Vp = (Vs0 + 1.164)/0.902
		rho=(310*(Vp*1000)**0.25)/1000
		vel_v[:,1]=np.abs(Vp)
		vel_v[:,3]=np.abs(rho)

		try:
			pd = GroupDispersion(*vel_v.T)
			cpr = pd(self.t, mode=0, wave="rayleigh") #Make sure to evaluate at same t you have disp curve measured at
			vo=cpr[1]
			to=cpr[0]

			plt.plot(to, vo, color="xkcd:purple", alpha=1.0, linewidth=2, label="Optimal solution")
		except:
			print("The dispersion curve did not converge!! Try varying the step number, or beta.")


		l1 = "L1=" + str(round(self.L1_norm(vel_v),1))
		plt.annotate(l1, xy=(1,1), xytext =(0.75,0.1), xycoords='axes fraction')
		plt.ylabel("Group V (km/s)")
		plt.xlabel("Period (s)")
		plt.legend(framealpha=0.15)

		plt.subplot(133)
		plt.plot(L1array)
		plt.xlabel("L1 norm")
		if (np.max(L1array) > 40):
			ylim = 40
			plt.ylim(0,ylim)

		plt.savefig("results.png")
		plt.show()
		return vel_olds, L1array








