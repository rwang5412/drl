from functools import partial
from importlib import import_module

# from env import CassieEnvClock

def env_factory(**kwargs):
    """Return callable object for env, so env is uninstantiated.
    """
    
    env_name = kwargs['env_name']
    env_module = import_module("env")
    
    try:
        # Assumes this particular env is imported inside ./env/__init__.py
        env = getattr(env_module, env_name)
        return partial(env, **kwargs)
    except:
        raise RuntimeError(f"Cannot locate env with name {env_name}.\n"
                           f"Check if it is imported properly inside /env/__init__.py")