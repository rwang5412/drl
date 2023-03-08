# Roadrunner Refactor

## Setup Instructions
Conda is required to run the setup script included with this repository. 
To install conda on your machine, visit [this page](https://docs.conda.io/en/latest/miniconda.html#linux-installers)
for download links.

To create a fresh conda env with all the necessary dependencies, simply run 
```
chmod +x setup.sh
source setup.sh
``` 
at the root directory of this repository. This script will setup a new conda env, install some additional pip packages, and finally install mujoco210. 

You should then be able to run the tests. Start with sim tests first, then env tests, then finally algo test. Run
```
python test.py --sim
python test.py --env
python test.py --algo
python test.py --nn
```
Note that for the sim test there is an intermittent seg fault issue with the libcassie viewer. If you get a segfault during libcassiesim test, you might have to try running it again a few times. We've found that it can sometimes happen if you close the viewer window too early.

To use roadrunner with the Duality submodule for any drake related functions, some additional setup steps are required. A git submodule is a link to a separate repository that is checked out at a certain commit. By default, `git clone` of this repo will only create an empty folder for the submodule without pulling its contents. 

To grab the contents of Duality, run 

```
git submodule update --init --recursive 
```

Submodules can also be updated to the latest version of the submodule's development branch. To receive submodule updates, run 

```
git submodule update --remote --merge
```

After setup, integration tests for Duality can be run using 

```
python test.py --drake
```