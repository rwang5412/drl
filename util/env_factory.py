from functools import partial
from importlib import import_module

def env_factory(**kwargs):
    """Return callable object for env, so env is uninstantiated.
    """

    env_name = kwargs['env_name']
    if "cassie" in env_name.lower():
        robot_type = "cassie"
    elif "digit" in env_name.lower():
        robot_type = "digit"
    else:
        raise RuntimeError("Please add options here to include a new robot type.")

    try:
        env_module = import_module(f"env.{robot_type}.{env_name.lower()}.{env_name.lower()}")
        env_class = getattr(env_module, env_name)
        return partial(env_class, **kwargs)
    except:
        raise RuntimeError(f"Cannot locate env with name {env_name}.\n"
                           f"Check if modules names are aligned exactly the same from class to folder names.")