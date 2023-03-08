import argparse
import json
import numpy as np

from env.genericenv import GenericEnv
from sim import MjDigitSim, ArDigitSim
from env.util.quaternion import (
    euler2quat,
    inverse_quaternion,
    rotate_by_quaternion,
    quaternion_product
)
from pathlib import Path

class DigitEnv(GenericEnv):
    def __init__(self,
                 simulator_type: str,
                 terrain: str,
                 policy_rate: int,
                 dynamics_randomization: bool,
                 state_noise: float):
        """Template class for Digit with common functions.
        This class intends to capture all signals under simulator rate (2kHz).

        Args:
            simulator_type (str): "mujoco" or "ar"
            clock (bool): "linear" or "von-Mises" or None
            policy_rate (int): Control frequency of the policy in Hertz
            dynamics_randomization (bool): True, enable dynamics randomization.
            terrain (str): Type of terrain generation [stone, stair, obstacle...]. Initialize inside
                           each subenv class to support individual use case.
        """
        super().__init__()
        self.dynamics_randomization = dynamics_randomization
        self.default_policy_rate = policy_rate
        self.terrain = terrain
        # Select simulator
        if simulator_type == "mujoco":
            self.sim = MjDigitSim()
        elif simulator_type == 'ar':
            self.sim = ArDigitSim()
        else:
            raise RuntimeError(f"Simulator type {simulator_type} not correct!"
                               "Select from 'mujoco' or 'ar'.")

        # Low-level control specifics
        self.offset = self.sim.reset_qpos[self.sim.motor_position_inds]
        self.kp = np.array([200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0,
                            200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0])
        self.kd = np.array([10.0, 10.0, 20.0, 20.0, 7.0, 7.0, 10.0, 10.0, 10.0, 10.0,
                            10.0, 10.0, 20.0, 20.0, 7.0, 7.0, 10.0, 10.0, 10.0, 10.0])

        # Init trackers to weigh/avg 2kHz signals and containers for each signal
        self.orient_add = 0
        self.trackers = [self.update_tracker_grf,
                         self.update_tracker_velocity]
        self.feet_grf_2khz_avg = {} # log GRFs in 2kHz
        self.feet_velocity_2khz_avg = {} # log feet velocity in 2kHz
        for foot in self.sim.feet_body_name:
            self.feet_grf_2khz_avg[foot] = self.sim.get_body_contact_force(name=foot)
            self.feet_velocity_2khz_avg[foot] = self.sim.get_body_velocity(name=foot)

        # Dynamics randomization ranges
        # If any joints/bodies are missing from the json file they just won't be randomized,
        # DR will still run. Get default ranges for each param too. We grab the indicies of the
        # relevant joints/bodies to avoid using named access later (vectorized access is faster)
        if self.__class__.__name__.lower() != "digitenv":
            dyn_rand_data = json.load(open(Path(__file__).parent /
                                        f"{self.__class__.__name__.lower()}/dynamics_randomization.json"))
            self.dr_ranges = {}
            # Damping
            damp_inds = []
            damp_ranges = []
            for joint_name, rand_range in dyn_rand_data["damping"].items():
                num_dof = len(self.sim.get_dof_damping(joint_name))
                for i in range(num_dof):
                    damp_inds.append(self.sim.get_joint_dof_adr(joint_name) + i)
                    damp_ranges.append(rand_range)
            damp_ranges = np.array(damp_ranges)
            self.dr_ranges["damping"] = {"inds":damp_inds,
                                        "ranges":damp_ranges}
            # Mass
            mass_inds = []
            mass_ranges = []
            for body_name, rand_range in dyn_rand_data["mass"].items():
                mass_inds.append(self.sim.get_body_adr(body_name))
                mass_ranges.append(rand_range)
            mass_ranges = np.array(mass_ranges)
            self.dr_ranges["mass"] = {"inds":mass_inds,
                                    "ranges":mass_ranges}
            # CoM location
            ipos_inds = []
            ipos_ranges = []
            for body_name, rand_range in dyn_rand_data["ipos"].items():
                ipos_inds.append(self.sim.get_body_adr(body_name))
                ipos_ranges.append(np.repeat(np.array(rand_range)[:, np.newaxis], 3, axis=1))
            ipos_ranges = np.array(ipos_ranges)
            self.dr_ranges["ipos"] = {"inds":ipos_inds,
                                    "ranges":ipos_ranges}
            # Friction
            self.dr_ranges["friction"] = {"ranges": dyn_rand_data["friction"]}
        self.state_noise = state_noise

        # Mirror indices and make sure complete test_mirror when changes made below
        # Readable string format listed in /testing/commmon.py
        # Digit's motor order is different between XML and Agility's header, here uses XML
        self.motor_mirror_indices = [-10, -11, -12, -13, -14, -15, -16, -17, -18, -19, # right leg/arm
                                     -0.1, -1, -2, -3, -4, -5, -6, -7, -8, -9          # left leg/arm
                                     ]
        # Proprioceptive state mirror inds should be synced up with get_robot_state()
        self.robot_state_mirror_indices = [0.01, -1, 2, -3,           # base orientation
                                        -4, 5, -6,                    # base rotational vel
                                        -17, -18, -19, -20, -21, -22, # right leg motor pos
                                        -23, -24, -25, -26,           # right arm motor pos
                                        -7,  -8,  -9,  -10, -11, -12, # left leg motor pos
                                        -13, -14, -15, -16,           # left arm motor pos
                                        -37, -38, -39, -40, -41, -42, # right leg motor vel
                                        -43, -44, -45, -46,           # right arm motor vel
                                        -27, -28, -29, -30, -31, -32, # left leg motor vel
                                        -33, -34, -35, -36,           # left arm motor vel
                                        -52, -53, -54, -55, -56,      # right joint pos
                                        -47, -48, -49, -50, -51,      # left joint pos
                                        -62, -63, -64, -65, -66,      # right joint vel
                                        -57, -58, -59, -60, -61,      # left joint vel
                                        ]

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
            f"State mirror inds size {len(self.get_observation_mirror_indices())} mismatch " \
            f"with observation size {self.observation_size}."
        assert len(self.get_action_mirror_indices()) == self.action_size, \
            "Action mirror inds size mismatch with action size."
