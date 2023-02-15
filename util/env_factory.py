import argparse

from functools import partial
from importlib import import_module
from types import SimpleNamespace
from util.colors import FAIL, ENDC

def env_factory(env_name: str, env_args):
    """Return callable object for env, so env is uninstantiated.
    """

    if "cassie" in env_name.lower():
        robot_type = "cassie"
    elif "digit" in env_name.lower():
        robot_type = "digit"
    else:
        raise RuntimeError("Please add options here to include a new robot type.")

    try:
        env_module = import_module(f"env.{robot_type}.{env_name.lower()}.{env_name.lower()}")
        env_class = getattr(env_module, env_name)
        if isinstance(env_args, list):
            env_argparse = argparse.ArgumentParser()
            env_argparse = env_module.add_env_args(env_argparse)
            env_args = env_argparse.parse_args(env_args)
        elif isinstance(env_args, SimpleNamespace) or isinstance(env_args, argparse.Namespace):
            env_args = env_module.add_env_args(env_args)
        else:
            raise RuntimeError(f"{FAIL}env_factory must receive either a list of un-parsed args " \
                               f"or a SimpleNamespace of already parsed args.{ENDC}")
        return partial(env_class, **vars(env_args))
    except:
        raise RuntimeError(f"Cannot locate env with name {env_name}.\n"
                           f"Check if modules names are aligned exactly the same from class to folder names.")