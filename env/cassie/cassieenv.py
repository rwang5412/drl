import numpy as np

from env import GenericEnv
from sim import MjCassieSim, LibCassieSim
from env.util.quaternion import (
    euler2quat,
    inverse_quaternion,
    rotate_by_quaternion,
    quaternion_product
)

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
        """
        super().__init__(policy_rate=policy_rate,
                         dynamics_randomization=dynamics_randomization)

        # Select simulator
        if simulator_type == "mujoco":
            self.sim = MjCassieSim()
            # Handle simulation features, such as heightmap
            if terrain:
                pass
        elif simulator_type == 'libcassie':
            self.sim = LibCassieSim()
            if terrain:
                pass
        else:
            raise RuntimeError(f"Simulator type {simulator_type} not correct!"
                               "Select from 'mujoco' or 'libcassie'.")

        # Generic env specifics
        self.orient_add = 0

        # Low-level control specifics
        self.offset = np.array([0.0045, 0.0, 0.4973, -1.1997, -1.5968, 0.0045, 0.0, 0.4973, -1.1997, -1.5968])
        self.kp = np.array([100,  100,  88,  96,  50, 100, 100,  88,  96,  50])
        self.kd = np.array([10.0, 10.0, 8.0, 9.6, 5.0, 10.0, 10.0, 8.0, 9.6, 5.0])

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
            self.foot_GRF = None

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
