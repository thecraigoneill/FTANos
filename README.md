# FTANos
FTAN analysis and inversion of seismic surface waves

## Installation

git clone https://github.com/thecraigoneill/FTANos \n
cd FTANos \n
pip install . \n

Make sure you have the requirements installed as noted in the requirements.txt file. We recommend an anaconda environment with disba and obspy installed. 

# Running FTANos
A self contained example from the paper can be run using the file run_FTAN.py.
It will create an FTAN map of the seismogram in the "example_trace.sac" file.

![Screen Shot 2023-11-09 at 12 43 54 pm](https://github.com/thecraigoneill/FTANos/assets/30849698/9c095fdd-d15f-4bc0-8d7f-5efd36552e2a)

The user then follows the prompts to click the origin, x max, and y max of the plot, before digitising the fundamental. Finish by pressing escape. 
The code then runs a MCMC inversion, which should give a result like this:

Although, products will differ due to the stochastic nature of the process. If the inversion fails or the fit is not acceptable (it happens), run again.



