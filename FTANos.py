"""
# FTANos core code May 2024. 
# Library for frequency-time analysis of surface wave dispersion.
# Routines for creation of FTAN maps, and inversion of FTAN curves.
# Developed by Craig O'Neill and Ao Chang, QUT. Acknowledge support of the GSQ.
# Distributed under MIT licence accompanying distribution.
# Accompanies paper "Utilising frequency-time analysis (FTAN) of surface waves for geotechnical and dam investigations"
"""

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
    """
    This base class contains the functionality to plot FTAN maps and digitise dispersion curves.
    The init function below defines a structure (self) that contains the variables used in this class.

    Parameters
    ----------

    self. fre1, self.fre2 : float
                                Lower and upper input frequencies for FTAN map range 

    self. vel1, self.vel2 : float
                                Lower and upper bounding velocities for FTAN map

    self.dist : float
                Distance between source and receiver for input trace (in m)

    self.alpha: float
                 Gaussian width of filter

    self.dt :  float
                Sampling interval of trace (in s)

    self.filename : string
                    Filename of SAC file containing the seismic trace. Uses obspy to read.

    self.st : Obspy stream
              Obspy formatted stream object of the imported sac file

    self.tr : Obspy trace
                Obspy formatted trace object of the important sac file - assuming a single trace (uses the first trace)

    self.x : 1D numpy float array
                Seismic trace in numpy array format, detrended. 

    self.dom : float
                Inverse of total time length.



    """
    def __init__(self, 
				 fre1 = 10,
				 fre2 = 100,
				 vel1 = 100,
				 vel2 = 1400,
				 dist = 30,
				 alpha = 25,
				 dt = 0.00025,
                 ftan_sc = 60,
				 filename = None,
                 im_file = None
					):

        self.fre1 = fre1
        self.fre2 = fre2
        self.T1 = 1/fre2
        self.T2 = 1/fre1
        self.vel1 = vel1
        self.vel2 = vel2
        self.alpha = alpha     
        self.dt = dt
        self.dist = dist
        self.ftan_sc = ftan_sc
        self.filename = filename
        self.st = read(self.filename, format="SAC")
        self.tr = self.st[0].detrend()
        self.x = self.tr.data
        self.dom = 1/(len(self.x)*self.dt)
        self.im_file = self.filename+".png"

    def times(self,):
        """
        # Returns a time array corresponding to the data array size (x), and the sampling rate dt.
        # These are imported in the structure self

        Parameters
        ----------

        self.x : 1D numpy float array
                 Data array of seismic trace in numpy array format

        self.dt :  float
                   Sampling rate of seismic data
 
        Returns
        -------

        t : 1D numpy float array
            An array of times corresponding to the length of the data trace    

        """
        t = np.arange(1e-12, np.size(self.x)*self.dt, self.dt)
        return t

    def periods(self,):
        """
        Simple routine to import 2 frequencies listed in the structure self, and
        exports an array of periods (n=40) between the corresponding periods

        Parameters
        ----------
        self. fre1, self.fre2 : float
                                Input frequencies (lower and higher)

        Returns
        -------
        p :  numpy float array
                array of periods for FTAN plotting
        """
        self.T1 = 1.0/self.fre2
        self.T2 = 1.0/self.fre1
        p = np.linspace(self.T1, self.T2, 40)
        return p

    def FTAN_a(self,):
        """
        FTAN routine for creation of FTAN maps. 
        Returns an amplitude map (2D array) of the dispersion response in period/freq, and group/phase velocity dimensions.
        Some parts cannibalised from CPS/AFTAN under BSD licence. Largely rewritten. 
        Some python inspired by other FTAN distros but again largely rewritten. 
        Inherits structure "self" containing time-series data x, sample rate dt, and periods.
        The FTAN also needs the band filter width alpha, which is predefined (or use the default)
        Ammplitude map scaled as per CPS and AFTAN. 

        Parameters
        ----------

        self.x :    Numpy float array
                    Data array of seismic trace in numpy array format

        self.periods :  Numpy float array
                        Array of periods to be used as x-axis in FTAN map

        self.dt :   Float
                    Sampling rate of seismic data

        self.alpha :  Float
                        Gaussian width of filter

        Returns
        -------

        amplitude :  2D numpy float array
                     2D numpy float array containing the FTAN amplitude spectrum (period vs v)

        """
        
        amplitude = np.zeros(shape=(len(self.periods()), len(self.x)))
        #apply Fourier transformation
        xi = fft(self.x)
        # array of frequencies
        freq = fftfreq(len(xi), d=self.dt)
 
        for iperiod, T0 in enumerate(self.periods()):
            f0 = 1.0/T0
            xi_f0 = xi*np.exp(-self.alpha*((freq-f0)/f0)**2) # Gaussian function around centre period
            #apply Fourier transformation back to time domain
            xi_f1 = ifft(xi_f0)/len(self.x)
            xi_f2 = np.copy(xi_f0)
            #filling amplitude and phase of column
            amplitude[iperiod, :] = self.ftan_sc* np.log10(np.abs(xi_f1)/(np.max(np.abs(xi_f1))-np.min(np.abs(xi_f0))))  # Normalising the amplitude range
            amax = -1.0e10
            for amp in amplitude[iperiod, :]:
                if amp > amax:
                    amax = amp
            amplitude[iperiod, :] = amplitude[iperiod, :] + 100.0 - amax  # Normalising the high amplitudes for clarity in dispersion curve
            i = 0
            for amp in amplitude[iperiod, :]:
                if amp < 40.0:
                    amplitude[iperiod, i] = 40.0 # Normalising the lower amplitudes for clarity in dispersion curve
                i += 1
            if iperiod == (len(self.periods())-1):
                print(np.c_[freq, amplitude[iperiod, :]])

            #phase[iperiod, :] = np.angle(xi_f1)  # We are not calculating phase information in this routine.
        return amplitude

    def plot_FTAN(self,):
        """
        # Function to create an FTAN plot
        # Regrids data onto a finer grid for presentation
        # Inherits structure "self", which has periods and v range, as well as amplitude from FTAN routine.
        # Creates a png FTAN map named " self.filename+".png" ", with filename being inherited from input.

        Parameters
        ----------

        self.FTAN_a :   Method
                        Call to method for creating FTAN map. Returns 2D numpy amplitude array.
        self.dist : float
                        Distance between source and receiver for input trace (in m)
        self.times : numpy float array
               Array of times corresponding to data trace 

        self.periods :  Numpy float array
                        Array of periods to be used as x-axis in FTAN map

        self.vel1, self.vel2 :  Numpy float array
                        Array of periods to be used as x-axis in FTAN map

        self.T1,self.T2 : Numpy floats
                        Values of max and min periods used in FTAN map

        self.filename : string
                    Filename of SAC file containing the seismic trace. Uses obspy to read.

        Results
        -------

        image file : png file
                    An image file in png format of the FTAN map
        """
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
        """
        # Function to create an FTAN plot. Inherits the structure "self".
        # Uses known range of periods and frequencies to scale mouse clicks on FTAN plot.
        # Creates a scaled dispersion curve, and saves it under self.filename+".disp"
        # Inherits a gtinker instance and digitises the plot.
        Usage:      outfile = FTANos(filename=sfile) 
                    outfile.plot_FTAN()
                    outfile.plot_digitise()

        Parameters
        ----------

        self.vel1, self.vel2 :  Numpy float array
                        Array of periods to be used as x-axis in FTAN map

        self.T1,self.T2 : Numpy floats
                        Values of max and min periods used in FTAN map

         self.filename : string
                    Filename of SAC file containing the seismic trace. Uses obspy to read.

        Results
        -------

        disp_file : ascii file
                    Ascii file of dispersion curve, in ( Period, Group velocity) format. Name is self.filename+".disp"


        """
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

    def plot_digitise_file(self,im_file):
        """
        # Function to create an FTAN plot from a file. Inherits the structure "self".
        # Uses known range of periods and frequencies to scale mouse clicks on FTAN plot.
        # Creates a scaled dispersion curve, and saves it under self.filename+".disp"
        # Loads a pre-created image file, and then creates a gtinker instance and digitises the plot.
        Usage:      outfile = FTANos(filename=sfile) 
                    outfile.plot_FTAN()
                    outfile.plot_digitise(im_file="my_ftan_map.png")

        Parameters
        ----------

        self.vel1, self.vel2 :  Numpy float array
                        Array of periods to be used as x-axis in FTAN map

        self.T1,self.T2 : Numpy floats
                        Values of max and min periods used in FTAN map

         self.filename : string
                    Filename of SAC file containing the seismic trace. Uses obspy to read.

        self.im_file : png image file
                    FTAN image file created from plot_FTAN()

        Results
        -------

        disp_file : ascii file
                    Ascii file of dispersion curve, in ( Period, Group velocity) format. Name is self.filename+".disp"


        """
        if (im_file):
            self.im_file = im_file
        print("Click points starting from [0,0] [0,1] [1,0]:") 
        print("First click the origin, maximum of x axis, and maximum of y-axis, in that order.")
        print("All clicks after the first three clicks are treated as data.")
        print("Hit escape to exit")
        # ####the user clicks over 3 times (must include [0,0], [0,1], [1,0]) on the figure and return the coordinates of each click in a list
        plt.clf()
        img = plt.imread(self.im_file)
        plt.imshow(img)
        plt.axis('image')
        x = plt.ginput(n=-1, timeout=0, show_clicks=True, mouse_stop="escape")
        plt.close()
        print("Pixel coordinates of the clicked points", np.array(x))
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


class FTAN_Invert:
    """
    This class hosts the inversion routines used to invert a dispersion curve to a shear wave velocity
    structure. Methods include Markov-Chain Monte Carlo, and a Nelder-Mead amoeba algorithm. It is recommended to 
    use NM first to converge on the solution space, then the MCMC with appropriate step sizes for the a priori
    distribution to cover the uncertainty space.
    
    The following initialisation routine generates a basic structure used in the inversion routines. Parameters are as
    follows:

    Parameters
    ----------

    alpha:  float
            Prefactor to MCMC step size

    beta:   float
            Exponent to MCMC step size (keep close to 1)

    step_size: float
                Size of MCMC inversion. Should be tuned to a priori distributions.

    step_floor: float
                Minimum step size to prevent over-reduction in stepsize through the parameter space

    n_burn: int
            Number of steps in the burn-in algorithm, if using raw MCMC

    n_ite:  int
            Number of steps in production MCMC run

    fig_file:   string
                Name of file for png output

    Poisson: float
            Used here to relate Vs to Vp and rho (density). The Vs inversion requires a velocity model
            with Vp and rho defined, and these should scale with Vs changes in a sensible way. 

    velMod: numpy float array
            Array containing the preliminary, and the updated velocity solution. Format is 
            "thickness, Vp, Vs, rho". Thickness refers to the thickness of each velocity layer. 
            Velocity in km/s, rho in g/cm3. Example format in initialisation below.

    velModa: numpy float array
             Spare array to store preliminary (or secondary) velocity models if required.

    Returns
    -------
    Initialisation returns the FTAN_invert structure for subsequent inversion routines.

    """
    def __init__(self,
				 alpha  = 0.5,		#Recommended 0.5
				 beta   = 1.02,		#Recommended 1.1		 
				 n_burn = 2000,
				 n_ite  = 5000,
				 fig_file = None,
				 step_size  = 0.25,	#Recommended 0.25 (there is a multiplier above)
				 step_floor = 0.08,  #Recommended 0.15 (150m/s)
                 Poisson = 0.4,
                 Vfac = 1.0,
                 max_vs = 4.0,
                 nm_step=0.11, 
                 nm_no_improve_thr=1e-6,
                 nm_no_improv_break=100, 
                 nm_max_iter=500,
                 nm_alpha=8., 
                 nm_gamma=6., 
                 nm_rho=0.55, 
                 nm_sigma=0.51,
                 nm_instances=18,
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
        self.Poisson = Poisson
        self.Vfac = Vfac
        self.max_vs = max_vs
        self.nm_step=nm_step 
        self.nm_no_improve_thr=nm_no_improve_thr                 
        self.nm_no_improv_break=nm_no_improv_break
        self.nm_max_iter=nm_max_iter                 
        self.nm_alpha=nm_alpha 
        self.nm_gamma=nm_gamma 
        self.nm_rho=nm_rho 
        self.nm_sigma=nm_sigma
        self.nm_instances = nm_instances
        self.t = np.genfromtxt(self.fig_file, usecols=0)
        self.disp_curve = np.genfromtxt(self.fig_file, usecols=1)/1e3

    #@jit(nopython=True)  # Uncomment if using jit here. Caution with some numpy routine compatibility.
    def L1_norm(self, vel):
        """
        Routine to calculate the L1 norm between the digitised dispersion curve, and a computed one in the inversion.
        Used by both MCMC and NelderMead inversion routines. Forward calculation uses disba routines. 

        Parameters
        ----------

        vel: numpy float array
             Preliminary velocity model used to calculate theoretical dispersion curve, using disba method "GroupDispersion"

        self.t: numpy float array (1D)
                Array of times associated with the digitised dispersion curve (self.disp_curve). Calculated dispersion is 
                interpolated to these times.

        self.disp_curve: numpy float array
                Group velocities of dispersion curve digitisation (from FTAN map). 

        Returns
        -------

        L1: float
            Numerical L1 norm value of fit between digitised and calculated dispersion curve.
        """
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
        L1 = np.sum(np.abs(self.disp_curve-v2)) + grad
        return L1

	#@jit(nopython=True)
	# def noise_deadener(self, Vs0a, noise):
    # """
    # Legacy routine
    # """
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
        """
        Routine to calculate the local average of a velocity structure.
        Used in the MCMC step determination.

        Parameters
        ----------

        n, self.n: int
                    Number of local layers to average

        a: numpy float array
            The array to calculate moving average on. Generally Vs here.

        Returns
        -------

        ret: numpy float array
             Array of moving average values.
            
        """
        self.n = n
        ret = np.cumsum(a, dtype=float)
        ret[self.n:] = ret[self.n:] - ret[:-self.n]
        return ret[self.n -1:]/self.n

    #@jit(nopython=True)
    def go_for_a_walk(self,):
        """
        # This is a Markov-Chain Monte Carlo approach for inverting the dispersion curve.
        # Routine inherits structure "self" including dispersion curve (self.disp_curve from FTAN), and initial velocity model (self.velMod)
        # Forward model uses disba libraries, which are a jit python distro based on the CPS forward modelling approach. 
        # Note we use an empirical scaling of Vp and rho from Vs to reduce the problem to Vs fit.
        # Produces a npy binary file of all post-burn MCMC instances.
        # Creates a file of dispersion results in ensemble median, stdv, and optimal fit solution
        # Also creates a png plot of the MCMC Vs profile results, as well as dispersion fit and L1 evolution. 
        # The disba routine can be sensitive to irregular Vs structures and may crash. 
        # Sometimes ensemble median solution also has no forward model solution and inversion fails.
        # If this happens, try running again (MCMC will vary each time, to a degree), or if it continues, 
        # you might need to vary the inversion parameters defined at the start of the problem. 
        # Inherits some numpy and scipy calls that are covered in the dependencies.

        Parameters
        ----------
        
        alpha:  float
            Prefactor to MCMC step size

        beta:   float
            Exponent to MCMC step size (keep close to 1)

        step_size: float
                Size of MCMC inversion. Should be tuned to a priori distributions.

        step_floor: float
                Minimum step size to prevent over-reduction in stepsize through the parameter space

        n_burn: int
            Number of steps in the burn-in algorithm, if using raw MCMC

        n_ite:  int
            Number of steps in production MCMC run

        fig_file:   string
                Name of file for png output

        Poisson: float
            Used here to relate Vs to Vp and rho (density). The Vs inversion requires a velocity model
            with Vp and rho defined, and these should scale with Vs changes in a sensible way. 

        velMod: numpy float array
            Array containing the preliminary, and the updated velocity solution. Format is 
            "thickness, Vs, Vp, rho". Thickness refers to the thickness of each velocity layer. 
            Velocity in km/s, rho in g/cm3. Example format in initialisation below.

        self.vel_orig : numpy float array
            Copy of the original velocities, for later plotting.

        velModa: numpy float array
             Spare array to store preliminary (or secondary) velocity models if required.

        self.moving_average : function
                See moving average function description

        self.L1_norm : function
                See L1_norm function description

        self.t, self.disp_curve : numpy float arrays
                        Results of digitised dispersion curve (periods and velocities, respectively)

        self.nm* : floats
                Parameters for Nelder Mead simplex algorithm. See routine for description.

        Results
        -------

        vel_olds: numpy float array (2D)
                Array of full ensemble of MCMC results. Each Vs is stored as a unique entry for later post-processing etc. 

        L1array: numpy float array (1D)
                Array of L1 results during inversion. In same order as vel_olds for fit comparison
        
        results.png: PNG figure file
                A figure of the inversion results, MCMC Vs ensemble and optimal inversion result. 

        """
        Poisson = self.Poisson
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
                Vp = Vs * np.sqrt( (1-Poisson)/(0.5 - Poisson))
                ro= 1740 * (Vp/1000)**0.25
                velocity_model[:,1] = Vp
                velocity_model[:,3] = ro 


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
                Vp = Vs0a * np.sqrt( (1-Poisson)/(0.5 - Poisson))
                ro= 1740 * (Vp/1000)**0.25
                vel_v[:,1] = Vp
                vel_v[:,3] = ro 
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
        Vp = mean_Vs * np.sqrt( (1-Poisson)/(0.5 - Poisson))
        ro= 1740 * (Vp/1000)**0.25
        vel_v[:,1] = Vp
        vel_v[:,3] = ro 

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


        Poisson=0.4
        vel_v[:,2]=vel_lowest[:,2] 
        #Vs = velocity_model[:,2]
        Vp = vel_v[:,2] * np.sqrt( (1-Poisson)/(0.5 - Poisson))
        ro= 1740 * (Vp/1000)**0.25
        vel_v[:,1] = Vp
        vel_v[:,3] = ro 

        try:
            pd = GroupDispersion(*vel_v.T)
            cpr = pd(self.t, mode=0, wave="rayleigh") #Make sure to evaluate at same t you have disp curve measured at
            vo=cpr[1]
            to=cpr[0]

            plt.plot(to, vo, color="xkcd:purple", alpha=1.0, linewidth=2, label="Optimal solution")
            np.savetxt("mcmc_optimal_model_dispersion.dat",np.c_[to,vo])
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

        # Best Vs results: mean_Vs (ensenble), lowest_Vs, stdv. 
        return (vel_olds, L1array)

    def get_velmod(self,Vs):
        """
        Routine to determine a self consistent Velocity model from a single Vs profile instance.
        Calculates Vp and rho, from Vs and Poisson's ratio.

        Parameters
        ----------

        self.Poisson: float
                                Poisson's ratio

        self.velMod: numpy float array
                     Array holding Vs profile

        Returns
        -------

        velocity_model: numpy float array (2D)
                        Array holding velocity model information in "width, Vp, Vs, rho" format. See initialisation information.
        """
        Poisson = self.Poisson
        velocity_model = self.velMod
        Vp = Vs * np.sqrt( (1-Poisson)/(0.5 - Poisson))
        if ( (np.all(Vp) > 0) & (~np.isnan(np.any(Vp))) ):
            try:
                ro= 1740 * ((0.001+np.abs(Vp))/1000)**0.25
                #print("Got here, Vp:",Vp)
            except:
                print("Issue with Vp:",Vp)
                ro=1740*(np.ones_like(Vp)*0.001/1000)**0.25
        else:
            ro=1740*(np.ones_like(Vp)*0.001/1000)**0.25
        velocity_model[:,2] = Vs
        velocity_model[:,1] = Vp
        velocity_model[:,3] = ro
        return(velocity_model)


    def fit_disp(self,velMod):
        """
        Alternate fit routine for amoeba minimisation.
        Calculates L2 and adds a gradient penalty

        Parameters
        ---------
        velMod: numpy float array
            Input velocity model in standard format (see initialisation for details)

        self.disp_curve, self.t : numpy float arrays
            Digitised dispersion curve

        Results
        ------

        Returns objective function value consisting of sum of L2 and grad.

        """
        velocity_model = velMod #np.copy(self.velMod)
        #velocity_model[:,2] = np.copy(self.velMod[:,2])
        v=np.ones_like(self.disp_curve)
        try:
            pd = GroupDispersion(*velocity_model.T)
            cpr = pd(self.t, mode=0, wave="rayleigh") #Make sure to evaluate at same t you have disp curve measured at
            v=cpr[1]
            to=cpr[0]
            fm=interp1d(to,v,fill_value="extrapolate")
            v2=fm(self.t)
        except:
            v2 = 1e9*(np.copy(self.disp_curve) + 0.5)
        h = velocity_model[:,0]
        #v3=np.copy(v2)
        try:
            diff = np.sqrt(np.mean((self.disp_curve - v2)**2)) #+ np.sum(np.gradients)
            diff2 = 6*np.sqrt(np.mean( (np.gradient(self.disp_curve) - np.gradient(v2))**2))
            #grad = 4.5*np.mean( np.abs( (np.roll(v3[1:]) - v3[1:])/h[1:]) )
            grad = 8.5*(np.mean(np.abs(np.gradient(v2))) )#*diff
        except:
            diff = 2* np.sqrt( (np.mean(self.disp_curve) - np.mean(v2) )**2)
            diff2 = np.sqrt(np.mean( (np.gradient(self.disp_curve) - np.gradient(v2))**2))
            grad = 5.5*(np.sum(np.abs(np.gradient(v2)))/np.size(v2) )#*diff
            print("BOOM")
        if (np.any(v2) > self.max_vs):
            diff *= 1e2
            diff2 *= 1e2
        #    print("Problem1")
        if (np.mean( np.gradient(v2) ) < 1e-6):
            diff *= 1e2
            diff2 *= 1e2
            grad*=1e2
        #    print("Problem2")
        if (len(v2) < 0.5*len(self.disp_curve)):
            diff *=1e3
            diff2 *=1e3
        #    print("Problem3")
        return(diff+diff2+grad)

    #@jit(nopython=False) 
    def nelder_mead(self):
        """
        Inherited version of standard amoeba minimisation from Nelder and Mead.
        Follows scipy's minimisation library approach largely, but internally modified
        to hold the velocity information and scale arrays appropriately. 

        Parameters
        ----------

        f : function 
            Function to optimize, must return a scalar score
            and operate over a numpy array of the same dimensions as x_start
            Routine L1_norm is used internally here.

        x_start : numpy float array 
                initial position (initial Vs array here)

        step :  float 
                look-around radius in initial step

        no_improv_thr,  no_improv_break : float, int
                break after no_improv_break iterations with
                an improvement lower than no_improv_thr

        max_iter : int 
                Always break after this number of iterations.
                Set it to 0 to loop indefinitely.

        alpha, gamma, rho, sigma : floats 
            parameters of the algorithm (see Wikipedia page for reference)
            Coefficients of reflection, expansion, contraction and shrinkage of the simplex
            Standard values are 1., 2., 0.5, 0.5.

        Returns
        -------

        res: tuple 
            best parameter array, best score

        """

        # init
        step=self.nm_step
        no_improve_thr=self.nm_no_improve_thr
        no_improv_break=self.nm_no_improv_break
        max_iter= self.nm_max_iter
        alpha=self.nm_alpha
        gamma=self.nm_gamma 
        rho=self.nm_rho 
        sigma=self.nm_sigma
        Vfac=self.Vfac
        vel_v = self.velMod
        Vs = np.copy(vel_v[:,2])
        x_start = Vs/Vfac

        dim = len(x_start)
        #prev_best = self.L1_norm(vel_v)
        prev_best = self.fit_disp(vel_v)
        no_improv = 0
        res = [[x_start, prev_best]]

        for i in range(dim):
            x = np.copy(x_start)
            x[i] = x[i] + step
            Vs =x*Vfac
            vMod = self.get_velmod(Vs)
            #score = self.L1_norm(vMod) #f(x)
            score = self.fit_disp(vMod) #f(x)
            res.append([x, score])

        # simplex iter
        iters = 0
        while 1:
            # order
            res.sort(key=lambda x: x[1])
            best = res[0][1]

            # break after max_iter
            if max_iter and iters >= max_iter:
                print("\t ... reached max iters")
                return res[0]
            iters += 1

            # break after no_improv_break iterations with no improvement
            #print( '...best so far:', best, res[0])

            if best < prev_best - no_improve_thr:
                no_improv = 0
                prev_best = best
            else:
                no_improv += 1

            if no_improv >= no_improv_break:
                print("\t ... no improvements break")
                return res[0]

            # centroid
            x0 = [0.] * dim
            for tup in res[:-1]:
                for i, c in enumerate(tup[0]):
                    x0[i] += c / (len(res)-1)

            # reflection
            xr = x0 + alpha*(x0 - res[-1][0])
            #self.Vs, self. h =xr
            Vs =xr*Vfac
            vMod = self.get_velmod(Vs)
            #rscore = self.L1_norm(vMod) #f(xr)
            rscore = self.fit_disp(vMod) #f(xr)
            if res[0][1] <= rscore < res[-2][1]:
                del res[-1]
                res.append([xr, rscore])
                continue

            # expansion
            if rscore < res[0][1]:
                xe = x0 + gamma*(x0 - res[-1][0])
                #self.Vs, self.h =xe
                Vs =xe*Vfac
                vMod = self.get_velmod(Vs)
                #escore = self.L1_norm(vMod) #f(xe)
                escore = self.fit_disp(vMod) #f(xe)
                if escore < rscore:
                    del res[-1]
                    res.append([xe, escore])
                    continue
                else:
                    del res[-1]
                    res.append([xr, rscore])
                    continue

            # contraction
            xc = x0 + rho*(x0 - res[-1][0])
            #self.Vs, self.h =xc
            Vs =xc*Vfac
            vMod = self.get_velmod(Vs)
            #cscore = self.L1_norm(vMod) #f(xc)
            cscore = self.fit_disp(vMod) #f(xc)
            if cscore < res[-1][1]:
                del res[-1]
                res.append([xc, cscore])
                continue

            # reduction
            x1 = res[0][0]
            nres = []
            for tup in res:
                redx = x1 + sigma*(tup[0] - x1)
                #self.Vs,self.h =redx
                Vs =redx*Vfac
                vMod = self.get_velmod(Vs)
                #score = self.L1_norm(vMod) #f(redx)
                score = self.fit_disp(vMod) #f(redx)
                nres.append([redx, score])
            res = nres

    def Amoeba_crawl(self,):
        """
        Routine to call amoeba NM method, and scale non-dimensionalised results back to Vs in km/s

        Parameters
        ----------
        self.Vfac : float
            Velocity scaling factor to non-dimensionalise array for minimisation(more useful when 
            minimising two different values at once, eg. Vs and h, but included for flexibility here)

        self.nelder_mead : function
            Traditional Nelder-Mead amoeba style minimisation. See function for details.

        self.nm_instances : int
            Define a number of intial conditions to begin the NM algorithm from, to effectively cover solution space.
            These ICs are arbitrarily scaled between 0.1 and 0.8 of the initial model.

        self.velMod : Numpy 2D float array
            Velocity model from input call, defined in standard CPS/Disba format.

        Returns
        -------

        ultimate_velMod : Numpy 2D float array
            Velocity model from NM inversion, defined in standard CPS/Disba format.

        ultimate_Vs: numpy float array
            Vs array that minimises calculated and digitised dispersion curves

        std: float
            Standard deviations of Vs model (following same layer format as Vs)
        
        best: numpy float array
            Array of best fit parameters from each inversion instance

        disp: tuple of numpy float arrays
            Contains calculated dispersion curve info in two tuple terms (periods in disp[0], group velocity in disp[1]

        disp_obs:  tuple of numpy float arrays
            Contains imported dispersion curve info in two tuple terms (periods in disp_obs[0], group velocity in disp_obs[1]
            
        """
        Vfac=self.Vfac
        Vs_original = np.copy(self.velMod[:,2])

        # Bounds of initial conditions
        # Assume runs from 5 - 50% of max Vs
        instances = np.linspace(0.1,0.8,self.nm_instances)
        i=0
        ultimate_best=1.0
        Vs = np.zeros( (np.size(instances),np.size(self.velMod[:,2])))
        Vsorig = np.zeros_like(Vs)
        best = np.zeros_like(instances)
        for f in instances:
            velModa = np.copy(self.velMod)
            velModa[:,2] = f*np.copy(Vs_original)
            velModa[10:,2] *=2
            velModa = self.get_velmod(velModa[:,2])

            self.velMod = velModa  #This will change the default original self.velMod
            Vsorig[i,:] = velModa[:,2]  #Initial conditions for inversion saved in this array

            result = self.nelder_mead()
            Vs[i,:] = result[0]*Vfac
            best[i] = result[1]
            velModb = self.get_velmod(Vs[i,:])
            
            if best[i] < ultimate_best:
                ultimate_best=best[i]
                #ultimate_disp0=disp[0]
                #ultimate_disp1=disp[1]
                ultimate_Vs = Vs[i,:]
                ultimate_velMod = np.copy(velModb)

            i += 1


        print("original conditions:",Vsorig)
        print("Amoeba results:",[Vs],"Best",best)

        std = np.std(Vs,axis=0)
        print("STD:",std)

        try:
            pd = GroupDispersion(*ultimate_velMod.T)
            cpr = pd(self.t, mode=0, wave="rayleigh")
            v1 = cpr[1]
            t1 = cpr[0]
            disp = (t1,v1)
        except:
            t1=self.t
            v1 = np.zeros_like(t1)
            disp=(t1,v1)
        disp_obs=(self.t,self.disp_curve)        

        return(ultimate_velMod, ultimate_Vs, std, best, disp, disp_obs)








