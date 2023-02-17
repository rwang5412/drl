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
from train import print_logo


if __name__ == "__main__":

    print_logo(subtitle="Maintained by Oregon State University's Dynamic Robotics Lab")

    # Setup arg Namespaces and get default values for algo args
    args = SimpleNamespace()
    env_args = SimpleNamespace()
    args = add_algo_args(args)

    # Overwrite with whatever optimization args you want here
    args.seed = 0
    args.traj_len = 200
    args.arch = "lstm"
    args.layers = "128,128"
    args.num_steps = 30000
    args.batch_size = 32
    args.epochs = 5
    args.dynamics_randomization = True
    args.discount = 0.95
    args.mirror = 0
    args.timesteps = 4e9
    args.workers = 56
    args.do_prenorm = False
    args.a_lr = 3e-4
    args.c_lr = 3e-4
    args.std = np.exp(-1.5)

    # Set env and logging args
    args.env_name = "CassieEnvClock"
    args.run_name = "CassieEnvClock_vonmises_test_seed{}".format(args.seed)
    args.logdir = "./logs/dump/"
    args.wandb = False
    args.nolog = False

    # Set env args
    env_args.simulator_type = "mujoco"
    env_args.terrain = False
    env_args.policy_rate = 50
    env_args.dynamics_randomization = True
    env_args.reward_name = "locomotion_vonmises_clock_reward"
    env_args.clock_type = "von_mises"

    run_experiment(args, env_args)
