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
    env_args = args_dict['env']
    non_env_args = args_dict['nonenv']

    # Resolve for actors from old roadrunner
    remove_keys = ["env_name", "calculate_norm"]
    for key in remove_keys:
        if key in actor_model_dict.keys():
            actor_model_dict.pop(key)
    if "fixed_std" in actor_model_dict.keys():
        actor_model_dict["learn_std"] = False if actor_model_dict['fixed_std'] is not None else True
        actor_model_dict.pop("fixed_std")

    # Load model class and checkpoint
    # args_dict.obs_dim = 43
    # args_dict.action_dim = 10
    # args_dict.layers = [64, 64]
    # from types import SimpleNamespace
    # env_args = SimpleNamespace()
    # env_args.simulator_type = "mujoco"
    # env_args.terrain = False
    # env_args.policy_rate = 50
    # env_args.dynamics_randomization = True
    # env_args.reward_name = "locomotion_linear_clock_reward"
    # env_args.clock_type = "linear"
    actor, critic = nn_factory(args=non_env_args)
    load_checkpoint(model=actor, model_dict=actor_model_dict)
    actor.eval()
    actor.training = False

    if evaluation_type == 'simple':
        simple_eval(actor=actor, env_name=non_env_args.env_name, args=env_args)
    else:
        raise RuntimeError(f"This evaluation type {evaluation_type} has not been implemented.")
