import json
import numpy as np
import os
from pathlib import Path
import traceback

from decimal import Decimal
from env.util.periodicclock import PeriodicClock
from env import DigitEnv
from importlib import import_module
from util.colors import *

class DigitEnvClock(DigitEnv):

    def __init__(self,
                 cycle_time: float,
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

        # Define env specifics
        self.observation_space = None
        self.action_space = None

        # Clock variables
        # NOTE: Both cycle_time and phase_add are in terms in raw time in seconds
        self.cycle_time = cycle_time
        phase_add = 1 / self.default_policy_rate
        swing_ratios = [0.4, 0.4]
        period_shifts = [0.0, 0.5]
        self.clock = PeriodicClock(self.cycle_time, phase_add, swing_ratios, period_shifts)
        self.clock_type = clock_type
        if self.clock_type == "von_mises":
            self.clock.precompute_von_mises()

        # Command variables
        self.traj_idx = 0
        self.orient_add = 0
        self.x_velocity = 0
        self.y_velocity = 0

        # Command randomization ranges
        self._x_velocity_bounds = [0.0, 3.0]
        self._y_velocity_bounds = [-0.3, 0.3]
        self._swing_ratio_bounds = [0.4, 0.8]
        self._period_shift_bounds = [0.0, 0.5]
        self._cycle_time_bounds = [0.75, 1.5]

        self.last_action = None

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
            print(f"{FAIL}ERROR: No such reward '{reward}'.{ENDC}")
            exit(1)
        except:
            print(traceback.format_exc())
            exit(1)

    def reset(self):
        """Reset simulator and env variables.

        Returns:
            state (np.ndarray): the s in (s, a, s')
        """
        self.reset_simulation()
        # Randomize commands
        self._x_velocity = np.random.uniform(*self._x_velocity_bounds)
        if self.x_velocity > 2.0:
            self.y_velocity = 0
        else:
            self.y_velocity = np.random.uniform(*self._y_velocity_bounds)
        swing_ratios = np.random.uniform(*self._swing_ratio_bounds, 2)
        period_shifts = np.random.uniform(*self._period_shift_bounds, 2)
        self.cycle_time = np.random.uniform(*self._cycle_time_bounds)
        phase_add = 1 / self.default_policy_rate
        # Update clock
        self.clock = PeriodicClock(self.cycle_time, phase_add, swing_ratios, period_shifts)
        if self.clock_type == "von_mises":
            self.clock.precompute_von_mises()
        self.traj_idx = 0
        self.orient_add = 0
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
        out = np.concatenate((self.get_robot_state(), [self.x_velocity, self.y_velocity],
                              self.clock.get_swing_ratios(), self.clock.get_period_shifts(),
                              self.clock.input_clock()))
        return out

    def get_action_mirror_indices(self):
        raise NotImplementedError

    def get_state_mirror_indices(self):
        raise NotImplementedError
