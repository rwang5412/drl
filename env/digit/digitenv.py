import json
import numpy as np

from env.genericenv import GenericEnv
from env.util.quaternion import *
from sim import MjDigitSim, ArDigitSim
from scipy.spatial.transform import Rotation as R
from util.colors import FAIL, WARNING, ENDC
from pathlib import Path
from testing.common import (
    DIGIT_JOINT_LLAPI2MJ_INDEX,
    DIGIT_MOTOR_LLAPI2MJ_INDEX,
)

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

        self.kp = np.array([80, 80, 110, 140, 40, 40, 80, 80, 50, 80,
                            80, 80, 110, 140, 40, 40, 80, 80, 50, 80])
        self.kd = np.array([8, 8, 10, 12, 6, 6, 9, 9, 7, 9,
                            8, 8, 10, 12, 6, 6, 9, 9, 7, 9])
        assert isinstance(state_noise, list), \
                f"{FAIL}Env {self.__class__.__name__} received 'state_noise' arg that was not a " \
                f"list. State noise must be given as a 6 long list.{ENDC}"
        assert len(state_noise) == 6, \
            f"{FAIL}Env {self.__class__.__name__} received 'state_noise' arg that was not 6 long. " \
            f"State noise must be given as a 6 long list.{ENDC}"
        if all(noise == 0 for noise in state_noise):
            self.state_noise = None
        else:
            self.state_noise = state_noise
        self.motor_encoder_noise = np.zeros(20)
        self.joint_encoder_noise = np.zeros(10)

        # Display menu of available commands for interactive control
        self._init_interactive_key_bindings()
        self.set_logging_fields()

        # Select simulator
        self.simulator_type = simulator_type
        if simulator_type == "mujoco":
            self.sim = MjDigitSim(terrain=terrain)
        elif simulator_type == 'ar_async':
            self.llapi_obs = None
            self.mj_sim = MjDigitSim(terrain=terrain)
            self.offset = self.mj_sim.reset_qpos[self.mj_sim.motor_position_inds]
            return
        else:
            raise RuntimeError(f"{FAIL}Simulator type {simulator_type} not correct!"
                               "Select from 'mujoco' or 'ar_async'.{ENDC}")

        # Low-level control specifics
        self.offset = self.sim.reset_qpos[self.sim.motor_position_inds]

        # Init trackers to weigh/avg high freq signals and containers for each signal
        self.orient_add = 0
        self.trackers = {self.update_tracker_grf: {"frequency": 50},
                         self.update_tracker_velocity: {"frequency": 50},
                         self.update_tracker_torque: {"frequency": 50},
                         self.update_tracker_cop: {"frequency": 50},
                        }
        # self.trackers[self.update_tracker_cop] = {"frequency": 50}
        # Double check tracker frequencies and convert to number of sim steps
        for tracker, tracker_dict in self.trackers.items():
            freq = tracker_dict["frequency"]
            steps = int(self.sim.simulator_rate // freq)
            if steps != self.sim.simulator_rate / freq:
                print(f"{WARNING}WARNING: tracker frequency for {tracker.__name__} of {freq}Hz " \
                      f"does not fit evenly into simulator rate of {self.sim.simulator_rate}. " \
                      f"Rounding to {self.sim.simulator_rate / steps:.0f}Hz instead.{ENDC}")
            tracker_dict["num_step"] = steps

        self.torque_tracker_avg = np.zeros(self.sim.num_actuators) # log torque in 2kHz
        self.feet_grf_tracker_avg = {} # log GRFs in 2kHz
        self.feet_velocity_tracker_avg = {} # log feet velocity in 2kHz
        for foot in self.sim.feet_body_name:
            self.feet_grf_tracker_avg[foot] = self.sim.get_body_contact_force(name=foot)
            self.feet_velocity_tracker_avg[foot] = self.sim.get_body_velocity(name=foot)
        self.cop = None
        self.cop_marker_id = None

        # Dynamics randomization ranges
        # If any joints/bodies are missing from the json file they just won't be randomized,
        # DR will still run. Get default ranges for each param too. We grab the indicies of the
        # relevant joints/bodies to avoid using named access later (vectorized access is faster)
        if self.__class__.__name__.lower() != "digitenv":
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
            # Spring stiffness
            spring_inds = []
            spring_ranges = []
            for joint_name, rand_range in dyn_rand_data["spring"].items():
                spring_inds.append(self.sim.get_joint_adr(joint_name))
                spring_ranges.append(rand_range)
            spring_ranges = np.array(spring_ranges)
            self.dr_ranges["spring"] = {"inds":spring_inds,
                                        "ranges":spring_ranges}
            # Friction
            self.dr_ranges["friction"] = {"ranges": dyn_rand_data["friction"]}
            self.dr_ranges["encoder-noise"] = {"ranges": dyn_rand_data["encoder-noise"]}
            dyn_rand_file.close()

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
        if self.simulator_type == "ar_async":
            raise RuntimeError(f"{FAIL}ERROR: 'reset_simulation' cannot be called when simulator "
                               f"type is ar_async.{ENDC}")

        if self.dynamics_randomization:
            self.sim.randomize_dynamics(self.dr_ranges)
            self.motor_encoder_noise = np.random.uniform(*self.dr_ranges["encoder-noise"]["ranges"], size=20)
            self.joint_encoder_noise = np.random.uniform(*self.dr_ranges["encoder-noise"]["ranges"], size=10)
            # NOTE: this creates very wrong floor slipperiness
            if self.terrain != "hfield":
                rand_euler = np.random.uniform(-.05, .05, size=2)
                rand_quat = scipy2mj(R.from_euler("xyz", [rand_euler[0], rand_euler[1], 0]).as_quat())
                self.sim.model.geom("floor").sameframe = 0
                self.sim.set_geom_quat("floor", rand_quat)
        else:
            self.sim.default_dynamics()
            self.motor_encoder_noise = np.zeros(20)
            self.joint_encoder_noise = np.zeros(10)
            if self.terrain != "hfield":
                self.sim.model.opt.disableflags = 0
                self.sim.model.geom("floor").sameframe = 1
                self.sim.set_geom_quat("floor", np.array([1, 0, 0, 0]))
        self.sim.reset()

    def step_simulation(self, action: np.ndarray, simulator_repeat_steps: int, integral_action: bool = False):
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
        if integral_action:
            setpoint = action + self.sim.get_motor_position()
        else:
            # Explore around neutral offset
            setpoint = action + self.offset
        # If using DR, need to subtract the motor encoder noise that we added in the robot_state
        if self.dynamics_randomization:
            setpoint -= self.motor_encoder_noise
        for sim_step in range(simulator_repeat_steps):
            # Send control setpoints and update torques
            self.sim.set_PD(setpoint=setpoint, velocity=np.zeros(action.shape), \
                            kp=self.kp, kd=self.kd)
            # step simulation
            self.sim.sim_forward()
            # Update simulation trackers (signals higher than policy rate, like GRF, etc)
            if sim_step > 0:
                for tracker_fn, tracker_dict in self.trackers.items():
                    if (sim_step + 1) % tracker_dict["num_step"] == 0 or sim_step + 1 == simulator_repeat_steps:
                        tracker_fn(weighting = 1 / np.ceil(simulator_repeat_steps / tracker_dict["num_step"]),
                                sim_step = sim_step)

    def get_robot_state(self):
        """Get standard robot prorioceptive states. Sub-env can override this function to define its
        own get_robot_state().

        Returns:
            robot_state (np.ndarray): robot state
        """
        if self.simulator_type == "ar_async":
            if self.llapi_obs is None:
                print(f"{WARNING}WARNING: llapi_obs is None, can not get robot state.{ENDC}")
                return False
            else:
                q = np.array([self.llapi_obs.base.orientation.w, self.llapi_obs.base.orientation.x,
                              self.llapi_obs.base.orientation.y, self.llapi_obs.base.orientation.z])
                base_ang_vel = np.array(self.llapi_obs.imu.angular_velocity[:])
                motor_pos = np.array(self.llapi_obs.motor.position[:])[DIGIT_MOTOR_LLAPI2MJ_INDEX]
                motor_vel = np.array(self.llapi_obs.motor.velocity[:])[DIGIT_MOTOR_LLAPI2MJ_INDEX]
                joint_pos = np.array(self.llapi_obs.joint.position[:])[DIGIT_JOINT_LLAPI2MJ_INDEX]
                joint_vel = np.array(self.llapi_obs.joint.velocity[:])[DIGIT_JOINT_LLAPI2MJ_INDEX]

        else:
            q = self.sim.get_base_orientation()
            # NOTE: do not use floating base angular velocity and it's bad on hardware
            base_ang_vel = self.sim.data.sensor('torso/base/imu-gyro').data
            motor_pos = self.sim.get_motor_position()
            motor_vel = self.sim.get_motor_velocity()
            joint_pos = self.sim.get_joint_position()
            joint_vel = self.sim.get_joint_velocity()

        base_orient = self.rotate_to_heading(q)

        # Add noise to motor and joint encoders per episode
        if self.dynamics_randomization:
            motor_pos += self.motor_encoder_noise
            joint_pos += self.joint_encoder_noise

        # Apply noise to proprioceptive states per step
        if isinstance(self.state_noise, list) and self.simulator_type != "ar_async":
            noise_euler = np.random.normal(0, self.state_noise[0], size = 3)
            noise_quat_add = R.from_euler('xyz', noise_euler)
            noise_quat = noise_quat_add * R.from_quat(mj2scipy(base_orient))
            base_orient = scipy2mj(noise_quat.as_quat())
            base_ang_vel = base_ang_vel + np.random.normal(0, self.state_noise[1], size = 3)
            motor_pos = motor_pos + np.random.normal(0, self.state_noise[2], size = self.sim.num_actuators)
            motor_vel = motor_vel + np.random.normal(0, self.state_noise[3], size = self.sim.num_actuators)
            joint_pos = joint_pos + np.random.normal(0, self.state_noise[4], size = self.sim.num_joints)
            joint_vel = joint_vel + np.random.normal(0, self.state_noise[5], size = self.sim.num_joints)

        robot_state = np.concatenate([
            base_orient,
            base_ang_vel,
            motor_pos,
            motor_vel,
            joint_pos,
            joint_vel
        ])
        return robot_state

    def update_tracker_grf(self, weighting: float, sim_step: int):
        """Keep track of 2khz signals, aggragate, and average uniformly.

        Args:
            weighting (float): weightings of each signal at simulation step to aggregate total
            sim_step (int): indicate which simulation step
        """
        for foot in self.feet_grf_tracker_avg.keys():
            if sim_step == 0: # reset at first sim step
                self.feet_grf_tracker_avg[foot] = np.zeros(3)
            else:
                self.feet_grf_tracker_avg[foot] += \
                    weighting * self.sim.get_body_contact_force(name=foot)

    def update_tracker_velocity(self, weighting: float, sim_step: int):
        for foot in self.feet_velocity_tracker_avg.keys():
            if sim_step == 0: # reset at first sim step
                self.feet_velocity_tracker_avg[foot] = np.zeros(6)
            else:
                self.feet_velocity_tracker_avg[foot] += \
                    weighting * self.sim.get_body_velocity(name=foot)

    def update_tracker_torque(self, weighting: float, sim_step: int):
        if sim_step == 0:   # reset at first sim step
            self.torque_tracker_avg = np.zeros(20)
        else:
            self.torque_tracker_avg += weighting * self.sim.get_torque()

    def update_tracker_cop(self, weighting: float, sim_step: int):
        """Keep track of 2khz signals, aggragate, and average uniformly.

        Args:
            weighting (float): weightings of each signal at simulation step to aggregate total
            sim_step (int): indicate which simulation step
        """
        if sim_step == 0:
            self.cop = None
        else:
            self.cop = self.sim.compute_cop()

    def rotate_to_heading(self, orientation: np.ndarray, hardware_imu: bool = False):
        """Offset robot heading in world frame by self.orient_add amount

        Args:
            orientation (list): current robot heading in world frame

        Returns:
            new_orient (list): Offset orientation
        """
        if hardware_imu:
            # Hardware LLAPI returns IMU orientation with 180 off in X
            quat = R.from_euler('xyz',[-np.pi,0,self.orient_add], degrees=False)
        else:
            quat = R.from_euler('xyz',[0,0,self.orient_add], degrees=False)
        new_quat = quat.inv() * R.from_quat(mj2scipy(orientation))
        q = scipy2mj(new_quat.as_quat())
        return q

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

    def _init_interactive_key_bindings(self):
        pass

    def set_logging_fields(self):
        # Define names for robot state input and action output
        self.output_names = ["left-hip-roll", "left-hip-yaw", "left-hip-pitch", "left-knee", "left-foot",
               "left-shoulder-roll", "left-shoulder-pitch", "left-shoulder-yaw", "left-elbow",
               "right-hip-roll", "right-hip-yaw", "right-hip-pitch", "right-knee", "right-foot",
               "right-shoulder-roll", "right-shoulder-pitch", "right-shoulder-yaw", "right-elbow"]
        self.robot_state_names = ["base-orientation-w", "base-orientation-x", "base-orientation-y", "base-orientation-z",
                            "base-roll-velocity", "base-pitch-velocity", "base-yaw-velocity",
                            "left-hip-roll-pos", "left-hip-yaw-pos", "left-hip-pitch-pos", "left-knee-pos", "left-foot-a-pos", "left-foot-b-pos",
                            "left-shoulder-roll-pos", "left-shoulder-pitch-pos", "left-shoulder-yaw-pos", "left-elbow-pos",
                            "right-hip-roll-pos", "right-hip-yaw-pos", "right-hip-pitch-pos", "right-knee-pos", "right-foot-a-pos", "right-foot-b-pos",
                            "right-shoulder-roll-pos", "right-shoulder-pitch-pos", "right-shoulder-yaw-pos", "right-elbow-pos",
                            "left-hip-roll-vel", "left-hip-yaw-vel", "left-hip-pitch-vel", "left-knee-vel", "left-foot-a-vel", "left-foot-b-vel",
                            "left-shoulder-roll-vel", "left-shoulder-pitch-vel", "left-shoulder-yaw-vel", "left-elbow-vel",
                            "right-hip-roll-vel", "right-hip-yaw-vel", "right-hip-pitch-vel", "right-knee-vel", "right-foot-a-vel", "right-foot-b-vel",
                            "right-shoulder-roll-vel", "right-shoulder-pitch-vel", "right-shoulder-yaw-vel", "right-elbow-vel",
                            "left-shin-pos", "left-tarsus-pos", "left-heel-spring-pos", "left-toe-pitch-pos", "left-toe-roll-pos",
                            "right-shin-pos", "right-tarsus-pos", "right-heel-spring-pos", "right-toe-pitch-pos", "right-toe-roll-pos",
                            "left-shin-vel", "left-tarsus-vel", "left-heel-spring-vel", "left-toe-pitch-vel", "left-toe-roll-vel",
                            "right-shin-vel", "right-tarsus-vel", "right-heel-spring-vel", "right-toe-pitch-vel", "right-toe-roll-vel"]