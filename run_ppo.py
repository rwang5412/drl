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

    parser = argparse.ArgumentParser()

    """All RL algorithms"""
    parser.add_argument("--seed",      default=0,           type=int)                  # random seed for reproducibility

    args = parser.parse_args()

    # Setup arg Namespaces and get default values for algo args
    # args = SimpleNamespace()
    env_args = SimpleNamespace()

    # Overwrite with whatever optimization args you want here
    args.seed = 0
    args.traj_len = 300
    args.arch = "lstm"
    args.layers = "128,128"
    args.num_steps = 30000
    args.batch_size = 32
    args.epochs = 5
    args.dynamics_randomization = True
    args.discount = 0.95
    args.mirror = 1
    args.timesteps = 4e9
    args.workers = 56
    args.do_prenorm = False
    args.a_lr = 3e-4
    args.c_lr = 3e-4
    args.std = np.exp(-1.5)

    # Set env and logging args
    args.env_name = "CassieEnvClockOld"
    args.run_name = f"CassieEnvClockOld_clockfix_test_seed{args.seed}"

    # Set env args
    args.simulator_type = "mujoco"
    args.terrain = False
    args.policy_rate = 50
    args.dynamics_randomization = True
    args.reward_name = "locomotion_linear_clock_reward"
    args.clock_type = "linear"

    args.wandb = True
    args.wandb_project_name = "roadrunner_refactor"
    args.logdir = "./logs/test/"
    # args.logdir = "./logs/dump/"
    # args.wandb = False

    args = add_algo_args(args)
    run_experiment(args, args.env_name)
