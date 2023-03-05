# Roadrunner Refactor

## Setup Instructions
A working conda environment is also provided and can be created from the included `spec-file.txt` (check the conda [guide](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#building-identical-conda-environments) for specific instructions). 

Conda is required to run the setup script included with this repository. To install conda on your machine, visit [this page](https://docs.conda.io/en/latest/miniconda.html#linux-installers) for download links. 

To create a fresh conda env with all the necessary dependencies, simply run 
```
source setup.sh
``` 
at the root directory of this repository. This script will setup a new conda env, install some additional pip packages, and finally install mujoco210. 

You should then be able to run the tests. Start with sim tests first, then env tests, then finally algo test. Run
```
python test.py --sim
python test.py --env
python test.py --algo
```
Note that for the sim test there is an intermittent seg fault issue with the libcassie viewer. If you get a segfault during libcassiesim test, you might have to try running it again a few times. We've found that it can sometimes happen if you close the viewer window too early.