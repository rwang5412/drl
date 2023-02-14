import argparse
import nn
import os
import pickle
import sys
import torch

from util.colors import OKGREEN, FAIL, ENDC
from util.env_factory import env_factory
from util.log import create_logger
from types import SimpleNamespace


def test_all_algos():
    test_ppo()
    print(f"{OKGREEN}Passed all algo tests{ENDC}")

def test_ppo():
    from algo.ppo import add_algo_args, run_experiment

    print("Testing PPO training")
    args = SimpleNamespace(env_name                 = "CassieEnvClock",
                           simulator_type           = "mujoco",
                           clock_type               = "linear",
                           reward_name              = "locomotion_linear_clock_reward",
                           policy_rate              = 50,
                           dynamics_randomization   = False,
                           terrain                  = False,
                           perception               = False,
                           logdir                   = "./trained_models/test/",
                           wandb                    = False,
                           nolog                    = True,
                           run_name                 = None,
                           seed                     = 0,
                           traj_len                 = 100,
                           timesteps                = 300,
                           prenormalize_steps       = 10,
                           num_steps                = 100,
                           discount                 = 0.99,
                           learn_stddev             = False,
                           std                      = 0.13,
                           a_lr                     = 1e-4,
                           c_lr                     = 1e-4,
                           eps                      = 1e-6,
                           kl                       = 0.02,
                           entropy_coeff            = 0.0,
                           clip                     = 0.05,
                           grad_clip                = 0.05,
                           batch_size               = 64,
                           epochs                   = 2,
                           mirror                   = 1,
                           do_prenorm               = True,
                           layers                   = "256,256",
                           arch                     = "ff",
                           bounded                  = False,
                           workers                  = 2,
                           redis                    = None,
                           previous                 = None)

    run_experiment(args)
    print(f"{OKGREEN}Passed PPO test{ENDC}")


