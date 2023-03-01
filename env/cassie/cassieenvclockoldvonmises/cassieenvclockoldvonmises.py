import argparse
import numpy as np
import os

from env.util.periodicclock import PeriodicClock
from env.cassie.cassieenvclock.cassieenvclock import CassieEnvClock
from types import SimpleNamespace
from util.colors import FAIL, WARNING, ENDC

class CassieEnvClockOldVonMises(CassieEnvClock):

    def __init__(self,
                 clock_type: str,
                 reward_name: str,
                 simulator_type: str,
                 terrain: str,
                 policy_rate: int,
                 dynamics_randomization: bool):
        assert clock_type == "linear" or clock_type == "von_mises", \
            f"{FAIL}CassieEnvClockOld received invalid clock type {clock_type}. Only \"linear\" or " \
            f"\"von_mises\" are valid clock types.{ENDC}"

        super().__init__(clock_type=clock_type,
                         reward_name=reward_name,
                         simulator_type=simulator_type,
                         terrain=terrain,
                         policy_rate=policy_rate,
                         dynamics_randomization=dynamics_randomization)

        # Command randomization ranges
        self._x_velocity_bounds = [0.5, 1.5]
        self._y_velocity_bounds = [-0.2, 0.2]
        self._swing_ratio_bounds = [0.5, 0.65]
        self._cycle_time_bounds = [0.75, 1.0]

        self.reset()

        # Define env specifics after reset
        self.sim.kp = np.array([70,  70,  100,  100,  50, 70,  70,  100,  100,  50])
        self.sim.kd = np.array([7.0, 7.0, 8.0,  8.0, 5.0, 7.0, 7.0, 8.0,  8.0, 5.0])

        # Define env specifics after reset
        self.observation_size = len(self.get_robot_state())
        self.observation_size += 3 # XYYaw velocity command
        self.observation_size += 2 # swing ratio
        self.observation_size += 2 # input clock
        self.action_size = self.sim.num_actuators
        # Only check sizes if calling current class. If is child class, don't need to check
        if os.path.basename(__file__).split(".")[0] == self.__class__.__name__.lower():
            self.check_observation_action_size()

    def reset(self):
        """Reset simulator and env variables.

        Returns:
            state (np.ndarray): the s in (s, a, s')
        """
        self.reset_simulation()
        # Randomize commands
        self.x_velocity = np.random.uniform(*self._x_velocity_bounds)
        self.y_velocity = np.random.uniform(*self._y_velocity_bounds)
        self.orient_add = 0

        # Update clock
        # NOTE: Both cycle_time and phase_add are in terms in raw time in seconds
        ratio = np.random.uniform(*self._swing_ratio_bounds)
        swing_ratios = [1 - ratio, ratio]
        period_shifts = [0.0, 0.5]
        self.cycle_time = np.random.uniform(*self._cycle_time_bounds)
        phase_add = 1 / self.default_policy_rate
        self.clock = PeriodicClock(self.cycle_time, phase_add, swing_ratios, period_shifts)
        if self.clock_type == "von_mises":
            self.clock.precompute_von_mises()

        # Reset env counter variables
        self.traj_idx = 0
        self.last_action = None
        return self.get_state()

    def get_state(self):
        out = np.concatenate((self.get_robot_state(),
                              self.clock.get_swing_ratios(),
                              [self.x_velocity, self.y_velocity, self.orient_add],
                              self.clock.input_sine_only_clock()))
        return out

    def get_observation_mirror_indices(self):
        mirror_inds = self.robot_state_mirror_indices
        # swing ratio
        mirror_inds += [len(mirror_inds) + 1, len(mirror_inds)]
        # XY Yaw velocity command
        mirror_inds += [len(mirror_inds), - (len(mirror_inds) + 1), - (len(mirror_inds) + 2)]
        # input clock sin
        mirror_inds += [len(mirror_inds) + 1, len(mirror_inds)]
        return mirror_inds

def add_env_args(parser: argparse.ArgumentParser | SimpleNamespace | argparse.Namespace):
    """
    Function to add handling of arguments relevant to this environment construction. Handles both
    the case where the input is an argument parser (in which case it will use `add_argument`) and
    the case where the input is just a Namespace (in which it will just add to the namespace with
    the default values) Note that arguments that already exist in the namespace will not be
    overwritten. To add new arguments if needed, they can just be added to the `args` dictionary
    which should map arguments to the tuple pair (default value, help string).

    Args:
        parser (argparse.ArgumentParser or SimpleNamespace, or argparse.Namespace): The argument
            parser or Namespace object to add arguments to

    Returns:
        argparse.ArgumentParser or SimpleNamespace, or argparse.Namespace: Returns the same object
            as the input but with added arguments.
    """
    args = {
        "simulator-type" : ("mujoco", "Which simulator to use (\"mujoco\" or \"libcassie\")"),
        "terrain" : ("", "What terrain to train with (default is flat terrain)"),
        "policy-rate" : (50, "Rate at which policy runs in Hz"),
        "dynamics-randomization" : (True, "Whether to use dynamics randomization or not (default is True)"),
        "reward-name" : ("locomotion_linear_clock_reward", "Which reward to use"),
        "clock-type" : ("linear", "Which clock to use (\"linear\" or \"von_mises\")")
    }
    if isinstance(parser, argparse.ArgumentParser):
        env_group = parser.add_argument_group("Env arguments")
        for arg, (default, help_str) in args.items():
            if isinstance(default, bool):   # Arg is bool, need action 'store_true' or 'store_false'
                env_group.add_argument("--" + arg, default = default, action = "store_" + \
                                    str(not default).lower(), help = help_str)
            else:
                env_group.add_argument("--" + arg, default = default, type = type(default), help = help_str)
    elif isinstance(parser, (SimpleNamespace, argparse.Namespace)):
        for arg, (default, help_str) in args.items():
            arg = arg.replace("-", "_")
            if not hasattr(parser, arg):
                setattr(parser, arg, default)
    else:
        raise RuntimeError(f"{FAIL}Environment add_env_args got invalid object type when trying " \
                           f"to add environment arguments. Input object should be either an " \
                           f"ArgumentParser or a SimpleNamespace.{ENDC}")

    return parser