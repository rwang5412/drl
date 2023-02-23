import argparse
import sys

from util.colors import BOLD, ORANGE, FAIL, ENDC

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    """Environment"""
    parser.add_argument("--env-name", default="CassieEnvClock", type=str, help="Name of environment to train on.")

    """Logger / Saver"""
    parser.add_argument("--wandb", default=False, action='store_true')
    parser.add_argument("--wandb-project-name", default="roadrunner_refactor")
    parser.add_argument("--logdir", default="./trained_models/", type=str)
    parser.add_argument("--run-name", default=None)

    """All RL algorithms"""
    parser.add_argument("--seed",      default=0, type=int)
    parser.add_argument("--traj-len",  default=300, type=int)
    parser.add_argument("--timesteps", default=1e8, type=float)

    assert len(sys.argv) >= 2, \
        f"{FAIL}Did not receive any arguments. Needs at least a \"algo\" argument. An example " \
        f"usage is\n`python train.py ppo [optional args]`{ENDC}"

    algo_name = sys.argv[1]
    sys.argv.remove(sys.argv[1])

    # Manually handle env-name argument
    try:
        env_name_idx = sys.argv.index("--env-name")
        env_name = sys.argv[env_name_idx + 1]
        if not isinstance(env_name, str):
            print(f"{__file__}: error: argument --env-name received non-string input.")
            sys.exit()
        # Delete env-name from command line input so isn't caught in parser below
        del sys.argv[env_name_idx:env_name_idx + 2]
    except ValueError:
        # If env-name not in command line input, use default value
        env_name = parser._option_string_actions["--env-name"].default

    # Select algorithm
    if algo_name == 'ppo':
        """
            Utility for running Proximal Policy Optimization.
        """
        from algo.ppo import add_algo_args, run_experiment

        parser = add_algo_args(parser)
        # args = parser.parse_args()
        # print(args)
        # exit()
        # Assume that any extra arguments will be handled later by the env arg parser in env_factory
        # args, env_args = parser.parse_known_args()
        run_experiment(parser, env_name)

    elif algo_name == 'diagnose':
        """
            Utility for diagonise training errors.
        """
        from util.check_number import unpack_training_error

        unpack_training_error('trained_models/CassieEnvClock/283a36-seed0/training_error.pt')