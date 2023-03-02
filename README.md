# Roadrunner Refactor

## Setup Instructions
A working conda environment is also provided and can be created from the included `spec-file.txt` (check the conda [guide](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#building-identical-conda-environments) for specific instructions). There are a few packages that are not included in conda and need to use pip to install instead. You also need to install the ar-control .whl file. After making your virtual env, activate it and in the repo home directory run:
```
pip install mujoco
pip install wandb
pip install ray==2.2.0
pip install setuptools==59.5.0
pip install ./sim/digit_sim/digit_ar_sim/agility-1.1.1-py3-none-any.whl
```

The libcassie sim also requires MuJoCo 2.10. Download mujoco tar [here](https://github.com/deepmind/mujoco/releases/tag/2.1.0) and extract it to a hidden folder in your home directory called `.mujoco`, such that the folder `~/.mujoco/mujoco210` exists.

You should then be able to run the tests. Start with sim tests first, then env tests, then finally algo test. Run
```
python test.py --sim
python test.py --env
python test.py --algo
```
Note that for the sim test there is an intermittent seg fault issue with the libcassie viewer. If you get a segfault during libcassiesim test, you might have to try running it again a few times. We've found that it can sometimes happen if you close the viewer window too early.