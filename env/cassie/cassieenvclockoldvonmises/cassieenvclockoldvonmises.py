import argparse
import numpy as np

from env.util.periodicclock import PeriodicClock
from env.cassie.cassieenvclock.cassieenvclock import CassieEnvClock
from types import SimpleNamespace
from util.colors import FAIL, WARNING, ENDC

class CassieEnvClockOldVonMises(CassieEnvClock):

    def __init__(self,
                 clock_type: str,
                 reward_name: str,
                 simulator_type: str,
                 terrain: bool,
                 policy_rate: int,
                 dynamics_randomization: bool,
                 **kwargs):
        assert clock_type == "linear" or clock_type == "von_mises", \
            f"{FAIL}CassieEnvClockOld received invalid clock type {clock_type}. Only \"linear\" or " \
            f"\"von_mises\" are valid clock types.{ENDC}"

        super().__init__(clock_type=clock_type,
                         reward_name=reward_name,
                         simulator_type=simulator_type,
                         terrain=terrain,
                         policy_rate=policy_rate,
                         dynamics_randomization=dynamics_randomization,
                         **kwargs)

        # Command randomization ranges
        self._x_velocity_bounds = [0.5, 1.5]
        self._y_velocity_bounds = [-0.2, 0.2]
        self._swing_ratio_bounds = [0.5, 0.65]
        self._cycle_time_bounds = [0.75, 1.0]

        self.reset()

        # Define env specifics after reset
        self.sim.kp = np.array([70,  70,  100,  100,  50, 70,  70,  100,  100,  50])
        self.sim.kd = np.array([7.0, 7.0, 8.0,  8.0, 5.0, 7.0, 7.0, 8.0,  8.0, 5.0])
        self.observation_size = len(self.get_state())
        self.action_size = self.sim.num_actuators

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
                              [self.x_velocity, 0, 0],
                              self.clock.input_sine_only_clock()))
        return out

def add_env_args(parser):
    args = {
        "simulator_type" : ("mujoco", "Which simulator to use (\"mujoco\" or \"libcassie\""),
        "perception" : (False, "Whether to use perception or not (default is False)"),
        "terrain" : (False, "What terrain to train with (default is flat terrain)"),
        "policy_rate" : (50, "Rate at which policy runs in Hz"),
        "dynamics_randomization" : (True, "Whether to use dynamics randomization or not (default is True)"),
        "reward_name" : ("locomotion_linear_clock_reward", "Which reward to use"),
        "clock_type" : ("linear", "Which clock to use (\"linear\" or \"von_mises\")")
    }
    if isinstance(parser, argparse.ArgumentParser):
        for arg, (default, help_str) in args.items():
            if isinstance(default, bool):   # Arg is bool, need action 'store_true' or 'store_false'
                parser.add_argument("--" + arg, default = default, action = "store_" + \
                                    str(not default).lower(), help = help_str)
            else:
                parser.add_argument("--" + arg, default = default, type = type(default), help = help_str)
    elif isinstance(parser, SimpleNamespace) or isinstance(parser, argparse.Namespace()):
        for arg, (default, help_str) in args.items():
            if not hasattr(parser, arg):
                setattr(parser, arg, default)
    else:
        raise RuntimeError(f"{FAIL}Environment add_env_args got invalid object type when trying " \
                           f"to add environment arguments. Input object should be either an " \
                           f"ArgumentParser or a SimpleNamespace.{ENDC}")

    return parser