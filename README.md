# Roadrunner Refactor

## Setup Instructions
Conda is required to run the setup script included with this repository.
To install conda on your machine, visit [this page](https://www.anaconda.com/download#downloads)
for download links.

To create a fresh conda env with all the necessary dependencies, simply run
```
chmod +x setup.sh
bash setup.sh
```
at the root directory of this repository. This script will setup a new conda env, install some additional pip packages, and install mujoco210.

You also need to have ar-software installed in order to run the Digit async simulation. Download it from the Google Drive [here](https://drive.google.com/file/d/1CbesyvdkN1s_V36csKk-cA5IdmaPGGlu/view?usp=sharing), unzip it and move it to your home directory.

You might need to install ffmpeg, with
```
sudo apt install ffmpeg
```

You should then be able to run the tests. Start with sim tests first, then env tests, then finally algo test. Run:
```
python test.py --all
```
Alternatively, you can run each test individually with the following commands:
```
python test.py --sim
python test.py --env
python test.py --algo
python test.py --nn
python test.py --render
python test.py --mirror
python test.py --timing
```
Note that for the sim test there is an intermittent seg fault issue with the libcassie viewer. If you get a segfault during libcassiesim test, you might have to try running it again a few times. We've found that it can sometimes happen if you close the viewer window too early.

## Developer Instructions
We currently are unable to have branch protection so be very careful to *NOT* push anything directly to develop. All changes to develop should be merged through PRs and have at least one reviewer approve. Branches should be named `feature` or `bugfix` accordingly, along with the feature/bugfix's corresponding Jira card number. For example, `feature/dp-xxx-my-new-feature`. If there is no such card what you are merging in that is fine and feel free to disregard the `dp-xxx-` part of the naming scheme.

When making a new feature to merge in, first switch to develop and pull, then make a new branch off of develop
```
git checkout develop
git pull
git checkout -b feature/dp-xxx-my-new-feature
```
Then make your commits on the branch as needed. Before merging the branch into develop (which should be done through a PR), make sure that branch is up to date with develop with
```
git fetch origin
git merge origin/develop
```
and resolve any conflicts if needed. You can add the flag `--no-commit` to prevent Git from commiting after doing the auto-merge so you can verify files yourself. I highly recommend VSCode's source control extension for double checking file changes before committing.

You can also have your own personal research branch, labelled `research/myname`. Feel free to have multiple personal research branches if needed, but I would like to keep the total number of branches down if possible, so try to delete any inactive branches you aren't using anymore.

Watch out for adding large files as well, this should be avoided at all times. Policies should not be added to repo, we instead suggest Google Drive/Dropbox as a cloud storage method for trained policies. Remember that vLab is *NOT* a reliable cloud storage place and should not be used as such. If there are important policies that you care about keeping, upload them to Google Drive/Dropbox, vLab may go out/lose data.

Policies will be added sparingly to the `./pretrained_models/` folder in develop for use in testing when needed, and in that case only the .pt files and experiment files will be added, no logging files.

Also note that this repo contains code provided by Agility Robotics that is intended for Digit customers *only*, so for now this repo is to remain private and any branches/forks of it should remain private as well.

## Training Instructions
Run the following to launch training on your local machine.
```
python run_ppo.py
```

You can also launch training from command with the `train.py` script. Algo generic arguments are defined in here while algo specific ones are defined in the corresponding algo file (see [`algo`](algo) for explanation on PPO arguments). `train.py` arguments are:
- `env-name`: The name of the environment to train on.
- `wandb`: Whether to use wandb for logging or not. By default is set to False.
- `wandb-project-name`: If using wandb, what is the project name to log to. By default is "roadrunner_refactor"
- `logdir`: The path of the directory to log and save files to.
- `run-name`: The name of the run/policy. Actor and critic `.pt` files, along with all logging files will be saved to the folder `logdir/env-name/run-name/timestamp/`. If a `run-name` is not provided, a hash string will be auto generated and used instead.
- `seed`: What to set the random seed to.
- `traj-len`: The maximum allowed trajectory length. During training, we will not collect samples beyond this point and will instead use the current critic to estimate the infinite horizon value.
- `timesteps`: How many total (for the *entire* training) timesteps to sample. Rather than train for a certain number of interations, we say that we are train using X amount of total samples. Often we just set this to be arbitrarily large like 5e9 to make the training run "forever" and then just stop the training manually ourselves when we see the learning curve plateau.

On vLab, go to `ppo_seeds.sh` and modify repo directory and conda env name, and then run
```
qsub ppo_seeds.sh
```
By default, this script will run the training using an exclusive node.

NOTE: If running on vLab you should set the wandb output directory (`--wandb-dir` if using `train.py` or `args.wandb_dir` if using `run_ppo.py`) to your own folder in tier1 storage (`/tier1/osu/username/somewhere`, you will need to make this directory if you haven't already, only `/tier1/osu/` is guaranteed to exist). If you don't do this, the output directory will be just the `./wandb` directory inside the repo, and it can get large very quickly. To prevent issues on vLab, it is recommended to keep the size of your home directory within 20GB. As such wandb should output to tier1 storage instead, where we have a shared 300GB limit. You can clean up the directory occasionally by using `wandb sync ./path/to/wandb/` to sync the runs, and then `wandb sync --clean ./path/to/wandb/` to delete old runs that have been already synced.

## Evaluation Instructions
After training a policy (or you can test with the provided policies in `./pretrained_models`) you can evaluate with the `eval.py` script. For example, run
```
python eval.py interactive --path ./pretrained_models/CassieEnvClock/spring_3_new/07-12-14-27/
```
to visualize and run a Cassie walking policy. Terminal printout will show a legend of keyboard commands along with what the current commands are. See `evaluation_factory` [documentation](util/readme.md#L15) for more details.

## Structure Overview

The repo is split into 6 main folders. Each contains it's own readme with further documentation.
- [`algo`](algo): Contains all of the PPO implentation code. All algorithm/training code should go here.
- [`nn`](nn): Contains all of the neural network definitions used for both actors and critics. Implements things like FF networks, LSTM networks, etc.
- [`env`](env): Contains all of the environment definitions. Split into Cassie and Digit envs. Also contains all of the reward functions.
- [`sim`](sim): Contains all of the simulation classes that the environment use and interact with.
- [`testing`](testing): Contains all of the testing functions used for CI and debugging. Performance testing for policies will go here as well.
- [`util`](util): Contains repo wide utility functions. Only utilities that are used across multiple of the above folders, or in scripts at the top level should be here. Otherwise they should go into the corresponding folder's util folder.
