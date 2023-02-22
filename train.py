import argparse
import nn
import os
import pickle
import sys
import torch

from algo.util.log import create_logger
from util.colors import BOLD, ORANGE, FAIL, ENDC
from util.env_factory import env_factory


def print_logo(subtitle=""):
    print()
    print(BOLD + ORANGE +  "                                                               ╱▔▔▔▔▔▔╲   ")
    print(BOLD + ORANGE +  "                                                             ▂╱ ╱▔╲    ╲  ")
    print(BOLD + ORANGE +  "                                                            ╱╲ ▕   ▏╲ ╲▕  ")
    print(BOLD + ORANGE +  "                                                           ╱ ▕▂▂▏  ╲╱▏▕▔  ")
    print(BOLD + ORANGE +  "                     _                                     ▏ ╱▂▂╲    ╲╱   ")
    print(BOLD + ORANGE +  "                    | |                                 ▕╲ ▋╱ ╱ ▕         ")
    print(BOLD + ORANGE +  " _ __ ___   __ _  __| |_ __ _   _ _ __  _ __   ___ _ __  ▏▔▔ ╱  ╱╲        ")
    print(BOLD + ORANGE +  "| '__/ _ \ / _` |/ _` | '__| | | | '_ \| '_ \ / _ \ '__| ╲▂▂╱▔▔▔╲ ╲       ")
    print(BOLD + ORANGE +  "| | | (_) | (_| | (_| | |  | |_| | | | | | | |  __/ |            ╲ ╲      ")
    print(BOLD + ORANGE +  "|_|  \___/ \__,_|\__,_|_|   \__,_|_| |_|_| |_|\___|_|             ╲ ╲     "+ ENDC)
    print("\n")
    print(subtitle)
    print("\n")

if __name__ == "__main__":

    print_logo(subtitle="Maintained by Oregon State University's Dynamic Robotics Lab")
    parser = argparse.ArgumentParser()

    """Environment"""
    parser.add_argument("--env-name",      default="CassieEnvClock", type=str)                     # environment to train on

    """Logger / Saver"""
    parser.add_argument("--wandb",    default=False, action='store_true')              # use weights and biases for training
    parser.add_argument("--wandb_project_name",    default="roadrunner_refactor")              # use weights and biases for training
    parser.add_argument("--logdir",   default="./trained_models/", type=str)
    parser.add_argument("--run_name", default=None)                                                 # run name

    """All RL algorithms"""
    parser.add_argument("--seed",      default=0,           type=int)                  # random seed for reproducibility
    parser.add_argument("--traj_len",  default=300,        type=int)                  # max trajectory length for environment
    parser.add_argument("--timesteps", default=1e8,         type=float)                # timesteps to run experiment for

    assert len(sys.argv) >= 2, \
        f"{FAIL}Did not receive any arguments. Needs at least a \"algo\" argument. An example " \
        f"usage is\n`python train.py ppo [optional args]`{ENDC}"

    algo = sys.argv[1]
    sys.argv.remove(sys.argv[1])

    if algo == 'ppo':
        """
            Utility for running Proximal Policy Optimization.

        """
        from algo.ppo import add_algo_args, run_experiment

        parser = add_algo_args(parser)
        # Assume that any extra arguments will be handled later by the env arg parser in env_factory
        args, env_args = parser.parse_known_args()
        run_experiment(args, env_args)

    elif algo == 'diagnose':
        """
            Utility for diagonise training errors.
        """
        from util.check_number import unpack_training_error

        unpack_training_error('trained_models/CassieEnvClock/283a36-seed0/training_error.pt')