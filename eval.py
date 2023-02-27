import torch
import argparse
import sys

from nn.actor import LSTMActor, FFActor
from util.colors import FAIL, ENDC, OKGREEN, WARNING
from util.env_factory import env_factory
from util.evaluation_factory import simple_eval

if __name__ == "__main__":

    """
    These parsers Env + NN will be removed and replaced by loading args from dict.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-name', default="CassieEnvClock", type=str)
    args, env_args = parser.parse_known_args()
    simple_eval(actor=None, env_name=args.env_name, args=env_args)
