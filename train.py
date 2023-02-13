import argparse
import nn
import os
import pickle
import sys
import torch

from util.colors import BOLD, ORANGE, ENDC
from util.env_factory import env_factory
from util.log import create_logger

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
    parser.add_argument("--not_dyn_random", dest='dynamics_randomization', default=True, action='store_false')
    parser.add_argument("--phase_std",  default=0.1, type=float)
    parser.add_argument("--task",       default='speed')
    parser.add_argument("--perception", default=False, action='store_true')
    parser.add_argument("--terrain",  default=False, action='store_true')
    parser.add_argument("--env-name",      default="CassieEnvClock", type=str)                     # environment to train on
    parser.add_argument("--reward-name", default="locomotion_linear_clock_reward", type=str)  # reward to use. this is a required argument.
    parser.add_argument("--simulator-type",   default="mujoco", type=str, help="Which simulatory to use")
    parser.add_argument("--clock-type",   default="linear", type=str, help="Which clock to use")
    parser.add_argument("--policy-rate",   default=50, type=int, help="Rate at which policy runs")

    """Logger / Saver"""
    parser.add_argument("--wandb",    default=False, action='store_true')              # use weights and biases for training
    parser.add_argument("--wandb_project_name",    default="roadrunner")              # use weights and biases for training
    parser.add_argument("--logdir",   default="./trained_models/", type=str)
    parser.add_argument("--run_name", default=None)                                                 # run name

    """All RL algorithms"""
    parser.add_argument("--nolog",     action='store_true')                            # store log data or not.
    parser.add_argument("--seed",      default=0,           type=int)                  # random seed for reproducibility
    parser.add_argument("--traj_len",  default=300,        type=int)                  # max trajectory length for environment
    parser.add_argument("--timesteps", default=1e8,         type=float)                # timesteps to run experiment for

    """
        Utility for running Proximal Policy Optimization.

    """
    from algo.ppo import run_experiment
    parser.add_argument("--prenormalize_steps", default=100,           type=int)
    parser.add_argument("--num_steps",          default=5000,          type=int)
    parser.add_argument('--discount',           default=0.99,          type=float)          # the discount factor
    parser.add_argument("--learn_stddev",       default=False,         action='store_true') # learn std_dev or keep it fixed
    parser.add_argument('--std',                default=0.13,          type=float)          # the fixed exploration std
    parser.add_argument("--a_lr",               default=1e-4,          type=float)          # adam learning rate for actor
    parser.add_argument("--c_lr",               default=1e-4,          type=float)          # adam learning rate for critic
    parser.add_argument("--eps",                default=1e-6,          type=float)          # adam eps
    parser.add_argument("--kl",                 default=0.02,          type=float)          # kl abort threshold
    parser.add_argument("--entropy_coeff",      default=0.0,           type=float)
    parser.add_argument("--clip",               default=0.2,           type=float)          # Clipping parameter for PPO surrogate loss
    parser.add_argument("--grad_clip",          default=0.05,          type=float)
    parser.add_argument("--batch_size",         default=64,            type=int)            # batch size for policy update
    parser.add_argument("--epochs",             default=3,             type=int)            # number of updates per iter
    parser.add_argument("--mirror",             default=0,             type=float)
    parser.add_argument("--do_prenorm",         default=False,         action='store_true') # Do pre-normalization or not


    parser.add_argument("--layers",             default="256,256",     type=str)            # hidden layer sizes in policy
    parser.add_argument("--arch",               default='ff')                               # either ff, lstm, or gru
    parser.add_argument("--bounded",            default=False,         type=bool)

    parser.add_argument("--workers",            default=2,             type=int)
    parser.add_argument("--redis",              default=None,          type=str)
    parser.add_argument("--previous",           default=None,          type=str)            # Dir of previously trained policy to start learning from

    args = parser.parse_args()


    run_experiment(args)
