import torch # first import to prevent ray 1 core bug
import numpy as np

from algo.ppo import add_algo_args, run_experiment
from types import SimpleNamespace

def run_ppo():
    # Setup arg Namespaces and get default values for algo args
    args = SimpleNamespace()

    # Overwrite with whatever optimization args you want here
    args.seed = np.random.randint(0, 100000)
    args.traj_len = 300
    args.arch = "lstm"
    args.layers = "64,64"
    args.num_steps = 50000
    args.batch_size = 32
    args.epochs = 2
    args.discount = 0.95
    args.gae_lambda = 0.95
    args.mirror = 1
    args.timesteps = 4e9
    args.workers = 80
    args.do_prenorm = False
    args.a_lr = 3e-4
    args.c_lr = 3e-4
    args.std = 0.13

    # Set env and logging args
    args.env_name = "CassieEnvClock"

    # Set env args
    args.simulator_type = "mujoco"
    args.state_est = False
    args.terrain = False
    args.policy_rate = 50
    args.dynamics_randomization = True
    args.reward_name = "locomotion_vonmises_clock_reward"
    args.clock_type = "von_mises"
    # args.state_noise = 0.0
    args.state_noise = [0.02, # orient noise (euler in rad)
                        0.03, # ang vel noise
                        0.01, # motor pos
                        0.03, # motor vel
                        0.01, # joint pos
                        0.03, # joint vel
                        ]
    args.full_clock = True
    args.integral_action = False

    args.run_name = f"feet"
    args.wandb = True
    args.wandb_project_name = "roadrunner_refactor"
    #NOTE: If running on vlab pur your tier1 folder here, ex. "/tier1/osu/username/"
    args.logdir = "/tier2/osu/dugarp/trained_models/"
    args.wandb_dir = "/tier2/osu/dugarp/"

    args = add_algo_args(args)
    run_experiment(args, args.env_name)
