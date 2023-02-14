import numpy as np

from env.util.periodicclock import PeriodicClock
from env.cassie.cassieenvclock.cassieenvclock import CassieEnvClock
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
