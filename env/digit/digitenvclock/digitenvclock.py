import argparse
import json
import numpy as np
from pathlib import Path
import traceback

from decimal import Decimal
from env.util.periodicclock import PeriodicClock
from env.digit.digitenv import DigitEnv
from importlib import import_module
from types import SimpleNamespace
from util.colors import FAIL, WARNING, ENDC
from util.check_number import is_variable_valid

class DigitEnvClock(DigitEnv):

    def __init__(self,
                 clock_type: str,
                 reward_name: str,
                 simulator_type: str,
                 terrain: bool,
                 policy_rate: int,
                 dynamics_randomization: bool):
        assert clock_type == "linear" or clock_type == "von_mises", \
            f"{FAIL}CassieEnvClock received invalid clock type {clock_type}. Only \"linear\" or " \
            f"\"von_mises\" are valid clock types.{ENDC}"

        super().__init__(simulator_type=simulator_type,
                         terrain=terrain,
                         policy_rate=policy_rate,
                         dynamics_randomization=dynamics_randomization)

        # Clock variables
        self.clock_type = clock_type

        # Command randomization ranges
        self._x_velocity_bounds = [0.0, 3.0]
        self._y_velocity_bounds = [-0.3, 0.3]
        self._swing_ratio_bounds = [0.4, 0.8]
        self._period_shift_bounds = [0.0, 0.5]
        self._cycle_time_bounds = [0.75, 1.5]

        # Load reward module
        self.reward_name = reward_name
        try:
            reward_module = import_module(f"env.rewards.{self.reward_name}.{self.reward_name}")
            reward_path = Path(__file__).parents[2] / "rewards" / self.reward_name / "reward_weight.json"
            self.reward_weight = json.load(open(reward_path))
            # Double check that reward weights add up to 1
            weight_sum = Decimal('0')
            for name, weight_dict in self.reward_weight.items():
                weighting = weight_dict["weighting"]
                weight_sum += Decimal(f"{weighting}")
            if weight_sum != 1:
                print(f"{WARNING}WARNING: Reward weightings do not sum up to 1, renormalizing.{ENDC}")
                for name, weight_dict in self.reward_weight.items():
                    weight_dict["weighting"] /= weight_sum
            self._compute_reward = reward_module.compute_reward
            self._compute_done = reward_module.compute_done
        except ModuleNotFoundError:
            print(f"{FAIL}ERROR: No such reward '{self.reward_name}'.{ENDC}")
            exit(1)
        except:
            print(traceback.format_exc())
            exit(1)

        self.reset()

        # Define env specifics after reset
        self.observation_size = len(self.get_robot_state()) # robot proprioceptive obs
        self.observation_size += 2 # XY velocity command
        self.observation_size += 2 # swing ratio
        self.observation_size += 2 # period shift
        self.observation_size += 2 # input clock
        self.action_size = self.sim.num_actuators
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
        swing_ratios = np.random.uniform(*self._swing_ratio_bounds, 2)
        period_shifts = np.random.uniform(*self._period_shift_bounds, 2)
        self.cycle_time = np.random.uniform(*self._cycle_time_bounds)
        phase_add = 1 / self.default_policy_rate
        self.clock = PeriodicClock(self.cycle_time, phase_add, swing_ratios, period_shifts)
        if self.clock_type == "von_mises":
            self.clock.precompute_von_mises()

        # Reset env counter variables
        self.traj_idx = 0
        self.last_action = None
        return self.get_state()

    def step(self, action: np.ndarray):
        if self.dynamics_randomization:
            self.policy_rate = self.default_policy_rate + np.random.randint(-5, 6)
        else:
            self.policy_rate = self.default_policy_rate

        # Step simulation by n steps. This call will update self.tracker_fn.
        simulator_repeat_steps = int(self.sim.simulator_rate / self.policy_rate)
        self.step_simulation(action, simulator_repeat_steps)
        self.clock.increment()
        # Reward for taking current action before changing quantities for new state
        r = self.compute_reward(action)

        self.traj_idx += 1
        self.last_action = action

        return self.get_state(), r, self.compute_done(), {}

    def compute_reward(self, action: np.ndarray):
        return self._compute_reward(self, action)

    def compute_done(self):
        return self._compute_done(self)

    def get_state(self):
        out = np.concatenate((self.get_robot_state(),
                              [self.x_velocity, self.y_velocity],
                              self.clock.get_swing_ratios(),
                              self.clock.get_period_shifts(),
                              self.clock.input_clock()))
        if not is_variable_valid(out):
            raise RuntimeError(f"States has Nan or Inf values. Training stopped.\n"
                               f"get_state returns {out}")
        return out

    def get_action_mirror_indices(self):
        return self.motor_mirror_indices

    def get_observation_mirror_indices(self):
        mirror_inds = self.robot_state_mirror_indices
        # XY velocity command
        mirror_inds += [len(mirror_inds), - (len(mirror_inds) + 1)]
        # swing ratio
        mirror_inds += [len(mirror_inds) + 1, len(mirror_inds)]
        # period shift
        mirror_inds += [- len(mirror_inds), - (len(mirror_inds) + 1)]
        # input clock sin/cos
        mirror_inds += [len(mirror_inds), len(mirror_inds) + 1]
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
        "simulator_type" : ("mujoco", "Which simulator to use (\"mujoco\" or \"libcassie\""),
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