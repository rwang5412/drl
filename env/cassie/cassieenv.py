import argparse
import numpy as np

from env.genericenv import GenericEnv
from sim import MjCassieSim, LibCassieSim
from env.util.quaternion import (
    euler2quat,
    inverse_quaternion,
    rotate_by_quaternion,
    quaternion_product
)
from util.colors import FAIL, ENDC

class CassieEnv(GenericEnv):
    def __init__(self,
                 simulator_type: str,
                 terrain: bool,
                 policy_rate: int,
                 dynamics_randomization: bool):
        """Template class for Cassie with common functions.
        This class intends to capture all signals under simulator rate (2kHz).

        Args:
            simulator_type (str): "mujoco" or "libcassie"
            clock (bool): "linear" or "von-Mises" or None
            policy_rate (int): Control frequency of the policy in Hertz
            dynamics_randomization (bool): True, enable dynamics randomization.
        """
        super().__init__()

        self.dynamics_randomization = dynamics_randomization
        self.default_policy_rate = policy_rate
        # Select simulator
        if simulator_type == "mujoco":
            self.sim = MjCassieSim()
        elif simulator_type == 'libcassie':
            self.sim = LibCassieSim()
        else:
            raise RuntimeError(f"Simulator type {simulator_type} not correct!"
                               "Select from 'mujoco' or 'libcassie'.")

        # Low-level control specifics
        self.offset = np.array([0.0045, 0.0, 0.4973, -1.1997, -1.5968, 0.0045, 0.0, 0.4973, -1.1997, -1.5968])
        self.kp = np.array([100,  100,  88,  96,  50, 100, 100,  88,  96,  50])
        self.kd = np.array([10.0, 10.0, 8.0, 9.6, 5.0, 10.0, 10.0, 8.0, 9.6, 5.0])

        # Init trackers to weigh/avg 2kHz signals and containers for each signal
        self.orient_add = 0
        self.trackers = [self.update_tracker_grf,
                         self.update_tracker_velocity]
        self.feet_grf_2khz_avg = {} # log GRFs in 2kHz
        self.feet_velocity_2khz_avg = {} # log feet velocity in 2kHz
        for foot in self.sim.feet_body_name:
            self.feet_grf_2khz_avg[foot] = self.sim.get_body_contact_force(name=foot)
            self.feet_velocity_2khz_avg[foot] = self.sim.get_body_velocity(name=foot)

        # Mirror indices and make sure complete test_mirror when changes made below
        # Readable string format listed in /testing/commmon.py
        self.motor_mirror_indices = [-5, -6, 7, 8, 9,
                                     -0.1, -1, 2, 3, 4]
        # Proprioceptive state mirror inds should be synced up with get_robot_state()
        self.robot_state_mirror_indices = [0.01, -1, 2, -3,      # base orientation
                                          -4, 5, -6,             # base rotational vel
                                          -12, -13, 14, 15, 16,  # right motor pos
                                          -7,  -8,  9,  10,  11, # left motor pos
                                          -22, -23, 24, 25, 26,  # right motor vel
                                          -17, -18, 19, 20, 21,  # left motor vel
                                          29, 30, 27, 28,        # joint pos
                                          33, 34, 31, 32, ]      # joint vel
        self.robot_state_feet_mirror_indices = [3, -4, 5,                # right foot position
                                                0.1, -1, 2,              # left foot position
                                                6, -7, 8, -9,            # base orientation
                                                -15, -16, 17, 18, 19,    # right motor pos
                                                -10, -11, 12, 13, 14,    # left motor pos
                                                20, -21, 22,             # base translational velocity
                                                -23, 24, -25,            # rotational velocity
                                                -31, -32, 33, 34, 35,    # right motor vel
                                                -26, -27, 28, 29, 30,    # left motor vel
                                                38, 39, 36, 37,          # joint pos
                                                42, 43, 40, 41]          # joint vel

    def reset_simulation(self):
        """Reset simulator.
        Depending on use cases, child class can override this as well.
        """
        self.sim.reset()

    def step_simulation(self, action: np.ndarray, simulator_repeat_steps: int):
        """This loop sends actions into control interfaces, update torques, simulate step,
        and update 2kHz simulation states.
        User should add any 2kHz signals inside this function as member variables and
        fetch them inside each specific env.

        Args:
            action (np.ndarray): Actions from policy inference.
        """
        for sim_step in range(simulator_repeat_steps):
            # Explore around neutral offset
            setpoint = action + self.offset
            # Send control setpoints and update torques
            self.sim.set_PD(setpoint=setpoint, velocity=np.zeros(action.shape), \
                            kp=self.kp, kd=self.kd)
            # step simulation
            self.sim.sim_forward()
            # Update simulation trackers (signals higher than policy rate, like GRF, etc)
            for tracker in self.trackers:
                tracker(weighting=1/simulator_repeat_steps, sim_step=sim_step)

    def get_robot_state(self):
        """Get standard robot prorioceptive states. Sub-env can override this function to define its
        own get_robot_state().

        Returns:
            robot_state (np.ndarray): robot state
        """
        robot_state = np.concatenate([
            self.rotate_to_heading(self.sim.get_base_orientation()),
            self.sim.get_base_angular_velocity(),
            self.sim.get_motor_position(),
            self.sim.get_motor_velocity(),
            self.sim.get_joint_position(),
            self.sim.get_joint_velocity()
        ])
        return robot_state

    def update_tracker_grf(self, weighting: float, sim_step: int):
        """Keep track of 2khz signals, aggragate, and average uniformly.

        Args:
            weighting (float): weightings of each signal at simulation step to aggregate total
            sim_step (int): indicate which simulation step
        """
        for foot in self.feet_grf_2khz_avg.keys():
            if sim_step == 0: # reset at first sim step
                self.feet_grf_2khz_avg[foot] = 0.0
            self.feet_grf_2khz_avg[foot] += \
                weighting * self.sim.get_body_contact_force(name=foot)

    def update_tracker_velocity(self, weighting: float, sim_step: int):
        for foot in self.feet_velocity_2khz_avg.keys():
            if sim_step == 0: # reset at first sim step
                self.feet_velocity_2khz_avg[foot] = 0.0
            self.feet_velocity_2khz_avg[foot] += \
                weighting * self.sim.get_body_velocity(name=foot)

    def rotate_to_heading(self, orientation: np.ndarray):
        """Offset robot heading in world frame by self.orient_add amount

        Args:
            orientation (list): current robot heading in world frame

        Returns:
            new_orient (list): Offset orientation
        """
        quaternion  = euler2quat(z=self.orient_add, y=0, x=0)
        iquaternion = inverse_quaternion(quaternion)

        if len(orientation) == 3:
            return rotate_by_quaternion(orientation, iquaternion)

        elif len(orientation) == 4:
            new_orient = quaternion_product(iquaternion, orientation)
            return new_orient

    def check_observation_action_size(self):
        """Check the size of observation/action/mirror. Subenv needs to define
        self.observation_size, self.action_size, self.get_state(),
        self.get_observation_mirror_indices(), self.get_action_mirror_indices().
        """
        assert self.observation_size == len(self.get_state()), \
            f"Check observation size = {self.observation_size}," \
            f"but get_state() returns with size {len(self.get_state())}"
        assert len(self.get_observation_mirror_indices()) == self.observation_size, \
            "State mirror inds size mismatch with observation size."
        assert len(self.get_action_mirror_indices()) == self.action_size, \
            "Action mirror inds size mismatch with action size."
