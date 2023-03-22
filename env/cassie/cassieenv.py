import argparse
import json
import numpy as np

from env.genericenv import GenericEnv
from sim import MjCassieSim, LibCassieSim
from env.util.quaternion import (
    euler2quat,
    inverse_quaternion,
    rotate_by_quaternion,
    quaternion_product
)
from util.colors import FAIL, WARNING, ENDC
from pathlib import Path

class CassieEnv(GenericEnv):
    def __init__(self,
                 simulator_type: str,
                 terrain: str,
                 policy_rate: int,
                 dynamics_randomization: bool,
                 state_noise: float,
                 state_est: bool):
        """Template class for Cassie with common functions.
        This class intends to capture all signals under simulator rate (2kHz).

        Args:
            simulator_type (str): "mujoco" or "libcassie"
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
        if state_est and not simulator_type == 'libcassie':
            raise RuntimeError(f"State estimator input can only be used with libcassie sim.")
        self.simulator_type = simulator_type
        if simulator_type == "mujoco":
            self.sim = MjCassieSim()
        elif simulator_type == 'libcassie':
            self.sim = LibCassieSim()
            self.state_est = state_est
        else:
            raise RuntimeError(f"Simulator type {simulator_type} not correct!"
                               "Select from 'mujoco' or 'libcassie'.")

        # Low-level control specifics
        self.offset = np.array([0.0045, 0.0, 0.4973, -1.1997, -1.5968, 0.0045, 0.0, 0.4973, -1.1997, -1.5968])
        self.kp = np.array([100,  100,  88,  96,  50, 100, 100,  88,  96,  50])
        self.kd = np.array([10.0, 10.0, 8.0, 9.6, 5.0, 10.0, 10.0, 8.0, 9.6, 5.0])

        # Init trackers to weigh/avg 2kHz signals and containers for each signal
        self.orient_add = 0
        self.trackers = {self.update_tracker_grf: {"frequency": 500},
                         self.update_tracker_velocity: {"frequency": 100}
                        }
        # Double check tracker frequencies and convert to number of sim steps
        for tracker, tracker_dict in self.trackers.items():
            freq = tracker_dict["frequency"]
            num_step = int(self.sim.simulator_rate // freq)
            if num_step != self.sim.simulator_rate / freq:
                print(f"{WARNING}WARNING: tracker frequency for {tracker.__name__} of {freq}Hz " \
                      f"does not fit evenly into simulator rate of {self.sim.simulator_rate}.{ENDC}")
            tracker_dict["num_step"] = num_step

        self.feet_grf_2khz_avg = {} # log GRFs in 2kHz
        self.feet_velocity_2khz_avg = {} # log feet velocity in 2kHz
        for foot in self.sim.feet_body_name:
            self.feet_grf_2khz_avg[foot] = self.sim.get_body_contact_force(name=foot)
            self.feet_velocity_2khz_avg[foot] = self.sim.get_body_velocity(name=foot)

        # Dynamics randomization ranges
        # If any joints/bodies are missing from the json file they just won't be randomized,
        # DR will still run. Get default ranges for each param too. We grab the indicies of the
        # relevant joints/bodies to avoid using named access later (vectorized access is faster)
        if self.__class__.__name__.lower() != "cassieenv":
            dyn_rand_file = open(Path(__file__).parent /
                                 f"{self.__class__.__name__.lower()}/dynamics_randomization.json")
            dyn_rand_data = json.load(dyn_rand_file)
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
            dyn_rand_file.close()
        self.state_noise = state_noise

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

        if self.dynamics_randomization:
            self.sim.randomize_dynamics(self.dr_ranges)
        else:
            self.sim.default_dynamics()

    def step_simulation(self, action: np.ndarray, simulator_repeat_steps: int):
        """This loop sends actions into control interfaces, update torques, simulate step,
        and update 2kHz simulation states.
        User should add any 2kHz signals inside this function as member variables and
        fetch them inside each specific env.

        Args:
            action (np.ndarray): Actions from policy inference.
        """
        # Reset trackers
        for tracker_fn, tracker_dict in self.trackers.items():
            tracker_fn(weighting = 0, sim_step = 0)
        for sim_step in range(simulator_repeat_steps):
            # Explore around neutral offset
            setpoint = action + self.offset
            # Send control setpoints and update torques
            self.sim.set_PD(setpoint=setpoint, velocity=np.zeros(action.shape), \
                            kp=self.kp, kd=self.kd)
            # step simulation
            self.sim.sim_forward()
            # Update simulation trackers (signals higher than policy rate, like GRF, etc)
            for tracker_fn, tracker_dict in self.trackers.items():
                if (sim_step + 1) % tracker_dict["num_step"] == 0:
                    tracker_fn(weighting = 1 / (simulator_repeat_steps / tracker_dict["num_step"]),
                               sim_step = sim_step)

    def get_robot_state(self):
        """Get standard robot prorioceptive states. Sub-env can override this function to define its
        own get_robot_state().

        Returns:
            robot_state (np.ndarray): robot state
        """
        if self.simulator_type == "libcassie" and self.state_est:
            robot_state = np.concatenate([
                self.rotate_to_heading(self.sim.get_base_orientation(state_est = self.state_est)),
                self.sim.get_base_angular_velocity(state_est = self.state_est),
                self.sim.get_motor_position(state_est = self.state_est),
                self.sim.get_motor_velocity(state_est = self.state_est),
                self.sim.get_joint_position(state_est = self.state_est),
                self.sim.get_joint_velocity(state_est = self.state_est)
            ])
        else:
            robot_state = np.concatenate([
                self.rotate_to_heading(self.sim.get_base_orientation()),
                self.sim.get_base_angular_velocity(),
                self.sim.get_motor_position(),
                self.sim.get_motor_velocity(),
                self.sim.get_joint_position(),
                self.sim.get_joint_velocity()
            ])
        robot_state += np.random.normal(0, self.state_noise, size = robot_state.shape)
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
