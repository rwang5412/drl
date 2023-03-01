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
        simple_eval(actor=None, env_name=args.env_name, env_args=env_args)
        exit()

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--path', default=None, type=str)
    # args = parser.parse_args()
    # model_path = args.path
    # previous_args_dict = pickle.load(open(model_path + "experiment.pkl", "rb"))
    # actor_checkpoint = torch.load(os.path.join(model_path, 'actor.pt'), map_location='cpu')

    # # Resolve for actors from old roadrunner
    # remove_keys = ["env_name", "calculate_norm"]
    # for key in remove_keys:
    #     if key in actor_checkpoint.keys():
    #         actor_checkpoint.pop(key)
    # if "fixed_std" in actor_checkpoint.keys():
    #     actor_checkpoint["learn_std"] = False if actor_checkpoint['fixed_std'] is not None else True
    #     actor_checkpoint.pop("fixed_std")

    # # Load model class and checkpoint
    # actor, critic = nn_factory(args=previous_args_dict['nn_args'])
    # load_checkpoint(model=actor, model_dict=actor_checkpoint)
    # actor.eval()
    # actor.training = False

    # if evaluation_type == 'simple':
    #     simple_eval(actor=actor, env_name=previous_args_dict['all_args'].env_name, args=previous_args_dict['env_args'])
    # else:
    #     raise RuntimeError(f"This evaluation type {evaluation_type} has not been implemented.")

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default=None, type=str)
    args = parser.parse_args()
    model_path = args.path
    # previous_args_dict = pickle.load(open(model_path + "experiment.pkl", "rb"))
    # actor_checkpoint = torch.load(os.path.join(model_path, 'actor.pt'), map_location='cpu')
    previous_args_dict = {'nn_args':None}
    actor_checkpoint = torch.load(os.path.join(model_path), map_location='cpu')

    # Resolve for actors from old roadrunner
    remove_keys = ["env_name", "calculate_norm"]
    for key in remove_keys:
        if key in actor_checkpoint.keys():
            actor_checkpoint.pop(key)
    if "fixed_std" in actor_checkpoint.keys():
        actor_checkpoint["learn_std"] = False if actor_checkpoint['fixed_std'] is not None else True
        actor_checkpoint.pop("fixed_std")

    # for key, val in actor_checkpoint['model_state_dict'].items():
    #     print(key, val.shape)
    # exit()
    # print(actor_checkpoint.keys())
    # exit()
    from nn.actor import FFActor, LSTMActor, GRUActor, MixActor

    # Load model class and checkpoint
    # actor, critic = nn_factory(args=previous_args_dict['nn_args'])
    # for key, val in actor.state_dict().items():
    #     print(key, val.shape)
    # exit()
    # actor = MixActor(obs_dim=42,
    #                     state_dim=35,
    #                     nonstate_dim=7,
    #                     action_dim=10,
    #                     lstm_layers=[64,64],
    #                     ff_layers=[64,64],
    #                     bounded=False,
    #                     learn_std=False,
    #                     std=0.13,
    #                     nonstate_encoder_dim=8,
    #                     nonstate_encoder_on=True)
    actor = MixActor(obs_dim=42,
                        state_dim=35,
                        nonstate_dim=7,
                        action_dim=11,
                        lstm_layers=[64,64],
                        ff_layers=[64,64],
                        bounded=False,
                        learn_std=False,
                        std=0.13,
                        nonstate_encoder_dim=8,
                        nonstate_encoder_on=False)
    for key, val in actor.state_dict().items():
        if 'lstm' in key or 'ff' in key:
            # print(key, val.shape)
            k = key.replace(".layers", "")
            actor.state_dict()[key].copy_(actor_checkpoint['model_state_dict'][k])
        if 'mean' in key:
            # print(key, val.shape)
            actor.state_dict()[key].copy_(actor_checkpoint['model_state_dict'][key])
        if 'nonstate' in key:
            # print(key, val.shape)
            k = key.replace(".layers", "")
            actor.state_dict()[key].copy_(actor_checkpoint['model_state_dict'][k])
    for key, val in actor_checkpoint.items():
        if hasattr(actor, key):
            # avoid loading private attributes
            if not key.startswith('_'):
                setattr(actor, key, val)
                # print(key, val)
    # load_checkpoint(model=actor, model_dict=actor_checkpoint)
    
    # for key, val in actor.state_dict().items():
    #     print(key, val)
    # exit()
    actor.eval()
    actor.training = False

    from types import SimpleNamespace
    env_args = SimpleNamespace()
    all_args = SimpleNamespace()
    # all_args.env_name = 'CassieEnvClockOldVonMises'
    all_args.env_name = 'CassieStone'
    previous_args_dict['env_args'] = env_args
    previous_args_dict['all_args'] = all_args

    if evaluation_type == 'simple':
        simple_eval(actor=actor, env_name=previous_args_dict['all_args'].env_name, env_args=previous_args_dict['env_args'])
    else:
        raise RuntimeError(f"This evaluation type {evaluation_type} has not been implemented.")
