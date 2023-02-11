import json
import numpy as np
import os
from pathlib import Path
import traceback

from decimal import Decimal
from env.util.periodicclock import PeriodicClock
from env.cassie.cassieenvclock.cassieenvclock import CassieEnvClock
from importlib import import_module
from util.colors import FAIL, WARNING, ENDC

class CassieEnvClockOld(CassieEnvClock):

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

        # self.reset()

    def reset(self):
        """Reset simulator and env variables.

        Returns:
            state (np.ndarray): the s in (s, a, s')
        """
        self.reset_simulation()
        # Randomize commands
        # NOTE: Both cycle_time and phase_add are in terms in raw time in seconds
        self.x_velocity = np.random.uniform(*self._x_velocity_bounds)
        if self.x_velocity > 2.0:
            self.y_velocity = 0
        else:
            self.y_velocity = np.random.uniform(*self._y_velocity_bounds)
        swing_ratios = [0.5, 0.5]#np.random.uniform(*self._swing_ratio_bounds, 2)
        period_shifts = [0.0, 0.5]#np.random.uniform(*self._period_shift_bounds, 2)
        self.cycle_time = 0.8#np.random.uniform(*self._cycle_time_bounds)
        phase_add = 1 / self.default_policy_rate
        if 1 < self.x_velocity <= 3:
            phase_add *= 1 + 0.5*(self.x_velocity - 1)/2
        elif self.x_velocity > 3:
            phase_add *= 1.5
        # Update clock
        self.clock = PeriodicClock(self.cycle_time, phase_add, swing_ratios, period_shifts)
        if self.clock_type == "von_mises":
            self.clock.precompute_von_mises()

        # Reset env counter variables
        self.traj_idx = 0
        self.orient_add = 0
        self.last_action = None
        return self.get_state()

    def get_state(self):
        out = np.concatenate((self.get_robot_state(),
                              self.clock.sin_input_clock(),
                              [self.x_velocity]))
        return out
