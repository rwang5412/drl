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

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default=None, type=str)
    args = parser.parse_args()
    model_path = args.path
    args_dict = pickle.load(open(model_path + "experiment.pkl", "rb"))
    actor_model_dict = torch.load(os.path.join(model_path, 'actor.pt'), map_location='cpu')

    # Resolve for actors from old roadrunner
    remove_keys = ["env_name", "calculate_norm"]
    for key in remove_keys:
        if key in actor_model_dict.keys():
            actor_model_dict.pop(key)
    if "fixed_std" in actor_model_dict.keys():
        actor_model_dict["learn_std"] = False if actor_model_dict['fixed_std'] is not None else True
        actor_model_dict.pop("fixed_std")

    # Load model class and checkpoint
    actor, critic = nn_factory(args=args_dict['nonenv'])
    load_checkpoint(model=actor, model_dict=actor_model_dict)
    actor.eval()
    actor.training = False

    if evaluation_type == 'simple':
        simple_eval(actor=actor, env_name=args_dict['nonenv'].env_name, args=args_dict['env'])
    else:
        raise RuntimeError(f"This evaluation type {evaluation_type} has not been implemented.")
