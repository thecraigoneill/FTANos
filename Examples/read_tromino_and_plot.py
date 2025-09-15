from FTANos import *


# This file shows two ways of creating an FTAn map and digitising the curve. 
# The default approach (1)  is to digitise a matplotlib instance.
# However, in some cases it us useful to revist an FTAN map and re-digitise, and we provide that option
# as a Figure loading example (2). 

# In either case, the first three clicks are expected to be the origin, the end of the x-axis, 
# and then the end of the yaxis. To reiterate:
# Click 1 = origin
# Click 2 = end of x-axis
# Click 3 = end of y-axis
# Make sure that the pixel coordinates are being displayed when clicking. If not, you may not be in a valid area
# of the plot. If the first three clicks aren't in this order, weird things can and will happen during the linear
# tranformation to real coordinates. 


#Input the filename of the seismic data
sfile = "Tromino_Eg.dat"

# One should always define these values. Those listed below are the defaults and will be used if no
# other information is given. 
# freq1 = 10
# freq2 = 500
# vel1 = 100
# vel2 = 1400
freq=1024
dt=1./freq
outfile = FTANos(filename=sfile,filetype="MOHO",dt=dt)# all the parameters listed can be changed according to the survey or set as default

# Here is an example where we set the parameters manually:
# outfile = FTANos(filename=sfile, fre1=freq1, fre2=freq2,vel1=vel1,vel2=vel2,dist=27.5, alpha=12.5, ftan_sc=60.0)#
# See the main codebase for full descriptions. The frequencies and velocities define the bounds of the FTAN map.
# Alpha is the gaussian kernel half-width for FTAN plotting. ftan_sc is a scaling term for plotting (from aftan).
# It is generally 60 but can be varied for different data noise levels. dist is distance between source and receiver.


# Plot FTAN map The following has two ways of digitising the FTAN map. The first creates the plot and digitises in one step. The second uses an existing FTAN map (here called "example_trace.sac.png"), and digitises that. This still assumes that the plot follows the frequency and velocity bounds defined in the previous step. 

# Option (1): create map and digitise directly in matplotlib:
outfile.plot_FTAN()

#Options (2): Import existing FTAN image file to digitise:
#outfile.plot_digitise_file(im_file="Tromino_Eg.dat.png")

# In either case, an output dispersion will be made, called "example_trace.sac.disp", 
# in a (periods (s), group velocity (m/s)) format.
