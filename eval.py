import torch
import argparse
import sys

from nn.actor import LSTMActor, FFActor
from util.colors import FAIL, ENDC, OKGREEN, WARNING
from util.env_factory import env_factory
from util.evaluation_factory import simple_eval

if __name__ == "__main__":

    # Choose which evaluation type
    try:
        evaluation_type = sys.argv[1]
        sys.argv = sys.argv[1:]
    except:
        raise RuntimeError("Need to choose evaluation type from ['simple','ui']. Or add a new evaluation type.")

    """
    These parsers Env + NN will be removed and replaced by loading args from dict.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default="./pretrained_models/lstm_speed_vonmises_fixedwalk_dx05to15.pt")
    parser.add_argument('--env-name', default="CassieEnvClockOldVonMises")
    args, env_args = parser.parse_known_args()

    # Make temp env to get input/output dimensions
    args, env_args = parser.parse_known_args()
    temp_env = env_factory(args.env_name, env_args)()

    actor_model_dict = torch.load(args.path, map_location='cpu')
    remove_keys = ["env_name", "calculate_norm"]
    for key in remove_keys:
        if key in actor_model_dict.keys():
            actor_model_dict.pop(key)
    if "fixed_std" in actor_model_dict.keys():
        actor_model_dict["learn_std"] = False if actor_model_dict['fixed_std'] is not None else True
        actor_model_dict.pop("fixed_std")
    if actor_model_dict["is_recurrent"]:
        actor = LSTMActor(input_dim=temp_env.observation_size,
                          action_dim=temp_env.action_size,
                          layers=[128,128],
                          bounded=False,
                          learn_std=False,
                          std=0.1)
    else:
        actor = FFActor(input_dim=temp_env.observation_size,
                        action_dim=temp_env.action_size,
                        layers=[256,256],
                        bounded=False,
                        learn_std=False,
                        std=0.1,
                        nonlinearity=torch.relu)
    # Create dict to check that all actor attributes are set
    actor_vars = set()
    for var in vars(actor):
        if var[0] != "_":
            actor_vars.add(var)
    for key, val in actor_model_dict.items():
        if key == "model_state_dict":
            actor.load_state_dict(val)
        elif hasattr(actor, key):
            setattr(actor, key, val)
        else:
            print(
                f"{FAIL}{key} in saved model dict, but actor {actor.__class__.__name__} has no such "
                f"attribute.{ENDC}")
        actor_vars.discard(key)
    # Double check that all actor attributes are set
    if len(actor_vars) != 0:
        miss_vars = ""
        for var in actor_vars:
            miss_vars += var + " "
        print(f"{WARNING}WARNING: Actor attribute(s) {miss_vars}were not set.{ENDC}")

    actor.eval()
    actor.training = False

    if evaluation_type == 'simple':
        simple_eval(actor=actor, env_name=args.env_name, args=env_args)
    else:
        raise RuntimeError(f"This evaluation type {evaluation_type} has not been implemented.")
