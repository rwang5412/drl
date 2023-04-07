import torch
import argparse
import sys
import pickle
import os

from util.evaluation_factory import simple_eval, eval_no_vis
from util.nn_factory import load_checkpoint, nn_factory
from util.env_factory import env_factory, add_env_parser

if __name__ == "__main__":

    try:
        evaluation_type = sys.argv[1]
        sys.argv.remove(sys.argv[1])
    except:
        raise RuntimeError("Choose evaluation type from ['simple','ui']. Or add a new one.")

    if evaluation_type == 'test':
        parser = argparse.ArgumentParser()
        parser.add_argument('--env-name', default="CassieEnvClock", type=str)
        # Manually handle env-name argument
        try:
            env_name_idx = sys.argv.index("--env-name")
            env_name = sys.argv[env_name_idx + 1]
            if not isinstance(env_name, str):
                print(f"{__file__}: error: argument --env-name received non-string input.")
                sys.exit()
        except ValueError:
            # If env-name not in command line input, use default value
            env_name = parser._option_string_actions["--env-name"].default
        add_env_parser(env_name, parser)
        args = parser.parse_args()
        for arg_group in parser._action_groups:
            if arg_group.title == "Env arguments":
                env_dict = {a.dest: getattr(args, a.dest, None) for a in arg_group._group_actions}
                env_args = argparse.Namespace(**env_dict)
        env = env_factory(env_name, env_args)()
        simple_eval(actor=None, env=env)
        exit()

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default=None, type=str)
    # Manually handle path argument
    try:
        path_idx = sys.argv.index("--path")
        model_path = sys.argv[path_idx + 1]
        if not isinstance(model_path, str):
            print(f"{__file__}: error: argument --path received non-string input.")
            sys.exit()
    except ValueError:
        print(f"No path input given. Usage is 'python eval.py simple --path /path/to/policy'")

    # model_path = args.path
    previous_args_dict = pickle.load(open(os.path.join(model_path, "experiment.pkl"), "rb"))
    actor_checkpoint = torch.load(os.path.join(model_path, 'actor.pt'), map_location='cpu')
    add_env_parser(previous_args_dict['all_args'].env_name, parser)
    args = parser.parse_args()
    # Overwrite previous env args with current input
    for arg, val in vars(args).items():
        if hasattr(previous_args_dict['env_args'], arg):
            setattr(previous_args_dict['env_args'], arg, val)

    # Load environment
    env = env_factory(previous_args_dict['all_args'].env_name, previous_args_dict['env_args'])()

    # Load model class and checkpoint
    actor, critic = nn_factory(args=previous_args_dict['nn_args'], env=env)
    load_checkpoint(model=actor, model_dict=actor_checkpoint)
    actor.eval()
    actor.training = False

    if evaluation_type == 'simple':
        simple_eval(actor=actor, env=env)
    elif evaluation_type == "no_vis":
        eval_no_vis(actor=actor, env=env)
    else:
        raise RuntimeError(f"This evaluation type {evaluation_type} has not been implemented.")
