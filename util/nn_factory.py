import argparse
import torch

from nn.critic import FFCritic, LSTMCritic, GRUCritic
from nn.actor import FFActor, LSTMActor, GRUActor, MixActor
from util.colors import FAIL, WARNING, ENDC

def nn_factory(args):
    """The nn_factory initializes a model class (actor, critic etc) by args (from saved pickle file
    or fresh new training). More cases can be added here to support different class types and init
    methods.

    Args:
        args (Namespace): Arguments for model class init.

    Returns: actor and critic
    """
    if args.arch == 'lstm':
        policy = LSTMActor(args.obs_dim,
                            args.action_dim,
                            std=args.std,
                            bounded=args.bounded,
                            layers=args.layers,
                            learn_std=args.learn_stddev)
        critic = LSTMCritic(args.obs_dim, layers=args.layers)
    elif args.arch == 'gru':
        policy = GRUActor(args.obs_dim,
                        args.action_dim,
                        std=args.std,
                        bounded=args.bounded,
                        layers=args.layers,
                        learn_std=args.learn_stddev)
        critic = GRUCritic(args.obs_dim, layers=args.layers)
    elif args.arch == 'ff':
        policy = FFActor(args.obs_dim,
                        args.action_dim,
                        std=args.std,
                        bounded=args.bounded,
                        layers=args.layers,
                        learn_std=args.learn_stddev,
                        nonlinearity=args.nonlinearity)
        critic = FFCritic(args.obs_dim, layers=args.layers)
    elif args.arch == 'mix':
        pass
    else:
        raise RuntimeError(f"Arch {args.arch} is not included, check the entry point.")

    return policy, critic

def load_checkpoint(model, model_dict: dict):
    """Load saved checkpoint (as dict) into a model definition. This process varies by use case ,
    but here tries to load all saved attributes from dict into the empty (or no-empty) model class.

    Args:
        model_dict (dict): A saved dict contains required attributes to initialize a model class.
        model: A model class, ie actor, critic, cnn etc. Thsi is not a direct nn.module, but a
               customized wrapper class with use-base dependent attributes.
    """
    # Create dict to check that all actor attributes are set
    model_vars = set()
    for var in vars(model):
        if var[0] != "_":
            model_vars.add(var)
    for key, val in model_dict.items():
        if key == "model_state_dict":
            model.load_state_dict(val)
        elif hasattr(model, key):
            if not key.startswith('_'): # avoid loading private attributes
                setattr(model, key, val)
        else:
            print(
                f"{FAIL}{key} in saved model dict, but model {model.__class__.__name__} has no such "
                f"attribute.{ENDC}")
        model_vars.discard(key)
    # Double check that all model attributes are set
    if len(model_vars) != 0:
        miss_vars = ""
        for var in model_vars:
            if not var.startswith('_'):
                miss_vars += var + " "
        print(f"{WARNING}WARNING: Model attribute(s) {miss_vars}were not set.{ENDC}")

def save_checkpoint(model, model_dict: dict, save_path: str):
    """Save a checkpoint by dict from a model class.

    Args:
        model: Any model class
        model_dict (dict): Saved dict.
        save_path (str): Saving path.
    """
    # Loop thru keys to make sure get any updates from model class
    # Excludes private attributes starting with "_"
    for key in vars(model):
        if not key.startswith('_'):
            model_dict[key] = getattr(model, key)
    torch.save(model_dict | {'model_state_dict': model.state_dict()},
                save_path)

def add_nn_parser(parser: argparse.ArgumentParser):
    nn_group = parser.add_argument_group("NN arguments")
    nn_group.add_argument("--std",      default=0.13, type=float, help="Action noise std dev")
    nn_group.add_argument("--bounded",  default=False, action="store_true",
                        help="Whether or not actor policy has bounded output")
    nn_group.add_argument("--layers", default="256,256", type=str,
                        help="Hidden layer size for actor and critic")
    nn_group.add_argument("--arch", default="ff", type=str,
                        help="Actor/critic NN architecture")
    nn_group.add_argument("--learn-stddev", default=False, action="store_true",
                        help="Whether or not to learn action std dev")
    nn_group.add_argument("--nonlinearity", default="tanh", type=str,
                        help="Actor output layer activation function")

    return parser