import torch
import argparse
import sys
import pickle
import os

from util.evaluation_factory import simple_eval
from util.nn_factory import load_checkpoint, nn_factory

if __name__ == "__main__":

    try:
        evaluation_type = sys.argv[1]
        sys.argv.remove(sys.argv[1])
    except:
        raise RuntimeError("Choose evaluation type from ['simple','ui']. Or add a new one.")

    if evaluation_type == 'test':
        parser = argparse.ArgumentParser()
        parser.add_argument('--env-name', default="CassieEnvClock", type=str)
        args, env_args = parser.parse_known_args()
        simple_eval(actor=None, env_name=args.env_name, args=env_args)
        exit()

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default=None, type=str)
    args = parser.parse_args()
    model_path = args.path
    previous_args_dict = pickle.load(open(model_path + "experiment.pkl", "rb"))
    actor_checkpoint = torch.load(os.path.join(model_path, 'actor.pt'), map_location='cpu')

    # Resolve for actors from old roadrunner
    remove_keys = ["env_name", "calculate_norm"]
    for key in remove_keys:
        if key in actor_checkpoint.keys():
            actor_checkpoint.pop(key)
    if "fixed_std" in actor_checkpoint.keys():
        actor_checkpoint["learn_std"] = False if actor_checkpoint['fixed_std'] is not None else True
        actor_checkpoint.pop("fixed_std")

    # Load model class and checkpoint
    actor, critic = nn_factory(args=previous_args_dict['nn_args'])
    load_checkpoint(model=actor, model_dict=actor_checkpoint)
    actor.eval()
    actor.training = False

    if evaluation_type == 'simple':
        simple_eval(actor=actor, env_name=previous_args_dict['all_args'].env_name, args=previous_args_dict['env_args'])
    else:
        raise RuntimeError(f"This evaluation type {evaluation_type} has not been implemented.")
