import torch
import argparse
import sys
import pickle
import os

from util.evaluation_factory import simple_eval
from util.nn_factory import load_checkpoint, nn_factory
from util.env_factory import env_factory

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
        simple_eval(actor=None, env_name=args.env_name, env_args=env_args)
        exit()

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default=None, type=str)
    args = parser.parse_args()
    model_path = args.path
    previous_args_dict = pickle.load(open(os.path.join(model_path, "experiment.pkl"), "rb"))
    actor_checkpoint = torch.load(os.path.join(model_path, 'actor.pt'), map_location='cpu')

    # Load environment
    env_fn = env_factory(previous_args_dict['all_args'].env_name, previous_args_dict['env_args'])

    # Load model class and checkpoint
    actor, critic = nn_factory(args=previous_args_dict['nn_args'], env_fn=env_fn)
    load_checkpoint(model=actor, model_dict=actor_checkpoint)
    actor.eval()
    actor.training = False

    if evaluation_type == 'simple':
        simple_eval(actor=actor, env_fn=env_fn)
    else:
        raise RuntimeError(f"This evaluation type {evaluation_type} has not been implemented.")
