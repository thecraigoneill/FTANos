This folder contains three example scripts, and three datasets. 

The script plot_digitise_FTAN.py is a python code to generate and digitise FTAN maps.
The script run_FTAN.py inverts the digitised inversion curves. 

The datafile example_trace.sac is a seismic trace in SAC format that can be read  by obspy. 
It is used by both plot_digitise_FTAN.py and run_FTAN.py scripts. 
The file example_trace.sac.disp is an inversion curve digitised by the script plot_digitise_FTAN.py, and can be used to replicate examples in the accompnaying paper. 

Finally, the script read_tromino_and_plot.py reads a Tromino MOHO instrument file (exported from the proprietry software Grilla into an ascii format, and called Tromino_Eg.dat). The method takes the Z column data and the sampling frequency as required input. It also reads the trigger column for active triggering events, and splices the time-series into discrete shots + 1 sec recordings. These traces are then stacked to improve signal to noise, and the resultant stack converted into a FTAN map. The map may then be use in plot_digitise_FTAN.py to construct a 1D velocity model for a site. Note this input format can be modified for any ascii input trace. 

See the Documentation directory for ipython notebooks/pdfs explaining these examples. 

