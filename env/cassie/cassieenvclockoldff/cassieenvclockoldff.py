import numpy as np

from env.util.periodicclock import PeriodicClock
from env.cassie.cassieenvclock.cassieenvclock import CassieEnvClock
from importlib import import_module
from util.colors import FAIL, WARNING, ENDC

class CassieEnvClockOldFF(CassieEnvClock):

    def __init__(self,
                 clock_type: str,
                 reward_name: str,
                 simulator_type: str,
                 terrain: bool,
                 policy_rate: int,
                 dynamics_randomization: bool,
                 **kwargs):
        assert clock_type == "linear" or clock_type == "von_mises", \
            f"{FAIL}CassieEnvClock received invalid clock type {clock_type}. Only \"linear\" or " \
            f"\"von_mises\" are valid clock types.{ENDC}"

        super().__init__(clock_type = clock_type,
                         reward_name = reward_name,
                         simulator_type=simulator_type,
                         terrain=terrain,
                         policy_rate=policy_rate,
                         dynamics_randomization=dynamics_randomization,
                         **kwargs)

        # Define env specifics
        self.sim.kp = np.array([80,  80,  88,  96,  50, 80,  80,  88,  96,  50])
        self.sim.kd = np.array([8.0, 8.0, 8.0, 9.6, 5.0, 8.0, 8.0, 8.0, 9.6, 5.0])

        self.reset()

        # Define env specifics after reset
        self.observation_space = len(self.get_state())
        self.action_space = self.sim.num_actuators

    def reset(self):
        """Reset simulator and env variables.

        Returns:
            state (np.ndarray): the s in (s, a, s')
        """
        self.reset_simulation()
        # Randomize commands
        # NOTE: Both cycle_time and phase_add are in terms in raw time in seconds
        self.x_velocity = 4.0#np.random.uniform(*self._x_velocity_bounds)
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

    def get_robot_state(self):
        motor_pos = self.sim.get_motor_position()
        joint_pos = self.sim.get_joint_position()

        motor_vel = self.sim.get_motor_velocity()
        joint_vel = self.sim.get_joint_velocity()

        base_pos = self.sim.get_base_position()
        l_foot_pos = self.sim.get_site_pose(self.sim.feet_site_name[0])[:3] - base_pos
        r_foot_pos = self.sim.get_site_pose(self.sim.feet_site_name[1])[:3] - base_pos
        # l_foot_pos = self.sim.get_body_pose(self.sim.feet_body_name[0])[:3] - base_pos
        # r_foot_pos = self.sim.get_body_pose(self.sim.feet_body_name[1])[:3] - base_pos

        # For feedforward policies, cannot internally estimate pelvis translational velocity, so need additional information
        # including foot position (relative to pelvis) and translational acceleration.
        # Can try inputting state estimated pelvis translational velocity, but this sometimes causes sim2real problems
        robot_state = np.concatenate([
            l_foot_pos,                                          # Left foot position
            r_foot_pos,                                         # Right foot position
            self.sim.get_base_orientation(),                 # Pelvis orientation
            motor_pos,                                                                      # Actuated joint positions
            self.rotate_to_heading(self.sim.get_base_linear_velocity()),       # Pelvis translational velocity
            self.sim.get_base_angular_velocity(),                                  # Pelvis rotational velocity
            motor_vel,                                                                      # Actuated joint velocities
            joint_pos,                                                                      # Unactuated joint positions
            joint_vel                                                                       # Unactuated joint velocities
        ])
        return robot_state

    def get_state(self):
        out = np.concatenate((self.get_robot_state(),
                              self.clock.input_sine_only_clock(),
                              [self.x_velocity]))
        return out
