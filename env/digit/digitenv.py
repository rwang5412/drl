import argparse
import numpy as np

from env.genericenv import GenericEnv
from sim import MjDigitSim, ArDigitSim
from env.util.quaternion import (
    euler2quat,
    inverse_quaternion,
    rotate_by_quaternion,
    quaternion_product
)
from types import SimpleNamespace

class DigitEnv(GenericEnv):
    def __init__(self,
                 simulator_type: str,
                 terrain: bool,
                 policy_rate: int,
                 dynamics_randomization: bool):
        """Template class for Digit with common functions.
        This class intends to capture all signals under simulator rate (2kHz).

        Args:
            simulator_type (str): "mujoco" or "ar"
            clock (bool): "linear" or "von-Mises" or None
        """
        super().__init__(policy_rate=policy_rate,
                         dynamics_randomization=dynamics_randomization)

        # Select simulator
        if simulator_type == "mujoco":
            self.sim = MjDigitSim()
            # Handle simulation features, such as heightmap
            if terrain:
                pass
        elif simulator_type == 'ar':
            self.sim = ArDigitSim()
            if terrain:
                pass
        else:
            raise RuntimeError(f"Simulator type {simulator_type} not correct!"
                               "Select from 'mujoco' or 'ar'.")

        # Generic env specifics
        self.orient_add = 0

        # Low-level control specifics
        self.reset_qpos = np.array([0, 0, 1,  1, 0 ,0 ,0,
        3.33020155e-01, -2.66178730e-02, 1.92369587e-01,
        9.93409734e-01, -1.04126145e-03, 1.82534311e-03, 1.14597921e-01,
        2.28971047e-01, 1.48527831e-03, -2.31455693e-01, 4.55857916e-04,
        -1.29734322e-02,
        9.89327705e-01, 1.45524756e-01, 1.73630859e-03, 7.08678995e-03,
        -2.03852305e-02,
        9.88035432e-01, 1.53876629e-01, 6.59769560e-05, 1.03905844e-02,
        -3.52778547e-03, -5.54992074e-02,
        -1.05542715e-01, 8.94852532e-01, -8.63756398e-03, 3.44780280e-01,
        -3.33020070e-01, 2.66195360e-02, -1.92382190e-01,
        9.93409659e-01, 1.04481446e-03, 1.82489637e-03, -1.14598546e-01,
        -2.28971188e-01, -1.48636971e-03, 2.31454977e-01, -4.53425792e-04,
        1.41299940e-02,
        9.89323654e-01, -1.45521550e-01, 1.80177609e-03, -7.67726135e-03,
        1.93767895e-02,
        9.88041418e-01, -1.53872399e-01, 1.26073837e-05, -9.87117790e-03,
        2.36874210e-03, 5.55559678e-02,
        1.05444698e-01, -8.94890429e-01, 8.85979401e-03, -3.44723293e-01
        ])

        self.offset = self.reset_qpos[self.sim.motor_position_inds]

        self.kp = np.array([200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0,
                            200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0])
        self.kd = np.array([10.0, 10.0, 20.0, 20.0, 7.0, 7.0, 10.0, 10.0, 10.0, 10.0,
                            10.0, 10.0, 20.0, 20.0, 7.0, 7.0, 10.0, 10.0, 10.0, 10.0])

        # Init trackers to weigh/avg 2kHz signals and containers for each signal
        self.trackers = [self.update_tracker_grf,
                         self.update_tracker_velocity]
        self.feet_grf_2khz_avg = {} # log GRFs in 2kHz
        self.feet_velocity_2khz_avg = {} # log feet velocity in 2kHz
        for foot in self.sim.feet_body_name:
            self.feet_grf_2khz_avg[foot] = self.sim.get_body_contact_force(name=foot)
            self.feet_velocity_2khz_avg[foot] = self.sim.get_body_velocity(name=foot)

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
        """Get standard robot prioceptive states

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

def add_env_args(parser):
    if isinstance(parser, argparse.ArgumentParser):
        parser.add_argument("--simulator-type",   default="mujoco", type=str, help="Which simulatory to \
                            use (\"mujoco\" or \"libcassie\"")
        parser.add_argument("--perception", default=False, action='store_true')
        parser.add_argument("--terrain",  default=False, action='store_true')
        parser.add_argument("--policy-rate",   default=50, type=int, help="Rate at which policy runs")
        parser.add_argument("--not_dyn_random", dest='dynamics_randomization', default=True, action='store_false')
        parser.add_argument("--reward-name", default="locomotion_linear_clock_reward", type=str)  # reward to use. this is a required argument.
        parser.add_argument("--clock-type",   default="linear", type=str, help="Which clock to use")
    elif isinstance(parser, SimpleNamespace) or isinstance(parser, argparse.Namespace()):
        default_values = {"simulator_type"          : "mujoco",
                          "perception"              : False,
                          "terrain"                 : False,
                          "policy_rate"             : 50,
                          "dynamics_randomization"  : False}
        for key, val in default_values.items():
            if not hasattr(parser, key):
                setattr(parser, key, val)
    else:
        raise RuntimeError(f"{FAIL}Environment add_env_args got invalid object type when trying " \
                           f"to add environment arguments. Input object should be either an " \
                           f"ArgumentParser or a SimpleNamespace.{ENDC}")

    return parser