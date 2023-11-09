# FTANos
FTAN analysis and inversion of seismic surface waves

## Installation

git clone https://github.com/thecraigoneill/FTANos

cd FTANos

pip install .


Make sure you have the requirements installed as noted in the requirements.txt file. We recommend an anaconda environment with disba and obspy installed. 

# Running FTANos
A self contained example from the paper can be run using the file run_FTAN.py.
It will create an FTAN map of the seismogram in the "example_trace.sac" file.

![Screen Shot 2023-11-09 at 12 43 54 pm](https://github.com/thecraigoneill/FTANos/assets/30849698/9c095fdd-d15f-4bc0-8d7f-5efd36552e2a)

The user then follows the prompts to click the origin, x max, and y max of the plot, before digitising the fundamental. Finish by pressing escape. 
The code then runs a MCMC inversion, which should give a result like this:

![Screen Shot 2023-11-09 at 1 12 20 pm](https://github.com/thecraigoneill/FTANos/assets/30849698/118a8c7f-c68b-44eb-8c6f-7e38f06c48d2)


Although, products will differ due to the stochastic nature of the process. If the inversion fails or the fit is not acceptable (it happens), run again.



