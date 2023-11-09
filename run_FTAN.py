
from FTAN_modules import *

#input the filename of the passive seismic data
sfile = "example_trace.sac"
# freq1 = 10
# freq2 = 500
# vel1 = 100
# vel2 = 1400
# T1 = 1.0/freq2
# T2 = 1.0/freq1
outfile = FTAN_modules(filename=sfile)# all the parameters listed can be changed according to the survey or set as default
# # # #plot FTAN map
outfile.plot_FTAN()
outfile.plot_digitise()

figname = sfile +".disp"
# # figname = "50m_nada.disp"
out_inv = MCMC(fig_file=figname)

vel_mod = out_inv.go_for_a_walk()
