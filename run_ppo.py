import argparse
import numpy as np
import os
import pickle
import sys

from algo.ppo import add_algo_args, run_experiment
from algo.util.log import create_logger
from util.colors import BOLD, ORANGE, FAIL, ENDC
from util.env_factory import env_factory
from types import SimpleNamespace

if __name__ == "__main__":

    # Setup arg Namespaces and get default values for algo args
    args = SimpleNamespace()
    env_args = SimpleNamespace()

    # Overwrite with whatever optimization args you want here
    args.seed = np.random.randint(0, 100000)
    args.traj_len = 300
    args.arch = "lstm"
    args.layers = "64,64"
    args.num_steps = 50000
    args.batch_size = 32
    args.epochs = 5
    args.discount = 0.95
    args.mirror = 1
    args.timesteps = 4e9
    args.workers = 56
    args.do_prenorm = False
    args.a_lr = 3e-4
    args.c_lr = 3e-4
    args.std = np.exp(-1.5)

    # Set env and logging args
    args.env_name = "CassieEnvClock"

    # Set env args
    args.simulator_type = "libcassie"
    args.terrain = False
    args.policy_rate = 50
    args.dynamics_randomization = False
    args.reward_name = "locomotion_vonmises_clock_reward"
    args.clock_type = "von_mises"
    args.state_noise = 0.0
    args.state_est = False

    args.run_name = f"{args.simulator_type}_{args.clock_type}_{args.arch}_nodr_noise"
    args.wandb = True
    args.wandb_project_name = "roadrunner_refactor"
    args.logdir = "./trained_models/"

    args = add_algo_args(args)
    run_experiment(args, args.env_name)
