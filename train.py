import argparse
import sys

from util.colors import BOLD, ORANGE, FAIL, ENDC

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    """Environment"""
    parser.add_argument("--env-name", default="CassieEnvClock", type=str)

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

    # Select algorithm
    if algo_name == 'ppo':
        """
            Utility for running Proximal Policy Optimization.
        """
        from algo.ppo import add_algo_args, run_experiment

        parser = add_algo_args(parser)
        # Assume that any extra arguments will be handled later by the env arg parser in env_factory
        args, env_args = parser.parse_known_args()
        run_experiment(args, env_args)

    elif algo_name == 'diagnose':
        """
            Utility for diagonise training errors.
        """
        from util.check_number import unpack_training_error

        unpack_training_error('trained_models/CassieEnvClock/283a36-seed0/training_error.pt')