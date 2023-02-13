import numpy as np

class GenericSim(object):
    """
    A base class to define the functions that interact with simulator.
    This class contains a set of getter/setter that unify generic naming conventions for different simulators
    """

    def __init__(self) -> None:
        pass

    def reset(self, qpos:list=None):
        raise NotImplementedError

    def sim_forward(self, dt: float = None):
        raise NotImplementedError

    def set_PD(self,
               setpoint: np.ndarray,
               velocity: np.ndarray,
               kp: np.ndarray,
               kd: np.ndarray):
        raise NotImplementedError

    def hold(self):
        raise NotImplementedError

    def release(self):
        raise NotImplementedError

    def viewer_init(self):
        raise NotImplementedError

    def viewer_draw(self):
        raise NotImplementedError

    """Getter/Setter to unify across simulators
    """
    def get_joint_position(self):
        raise NotImplementedError

    def set_joint_position(self, position: np.ndarray):
        raise NotImplementedError

    def get_joint_velocity(self):
        raise NotImplementedError

    def set_joint_velocity(self, velocity: np.ndarray):
        raise NotImplementedError

    def get_motor_position(self):
        raise NotImplementedError

    def set_motor_position(self, position: np.ndarray):
        raise NotImplementedError

    def get_motor_velocity(self):
        raise NotImplementedError

    def set_motor_velocity(self, velocity: np.ndarray):
        raise NotImplementedError

    def get_base_position(self):
        raise NotImplementedError

    def set_base_position(self, position: np.ndarray):
        raise NotImplementedError

    def get_base_linear_velocity(self):
        raise NotImplementedError

    def set_base_linear_velocity(self, velocity: np.ndarray):
        raise NotImplementedError

    def get_base_orientation(self):
        raise NotImplementedError

    def set_base_orientation(self, quat: np.ndarray):
        raise NotImplementedError

    def get_base_angular_velocity(self):
        raise NotImplementedError

    def set_base_angular_velocity(self, velocity: np.ndarray):
        raise NotImplementedError

    def get_torque(self):
        raise NotImplementedError

    def set_torque(self, torque: np.ndarray):
        raise NotImplementedError

    def get_joint_qpos_adr(self, name: str):
        raise NotImplementedError()

    def get_joint_dof_adr(self, name: str):
        raise NotImplementedError()

    def get_body_pose(self, name: str, relative_to_body_name=False):
        raise NotImplementedError

    def get_body_velocity(self, name: str, local_frame=True):
        raise NotImplementedError
    
    def get_body_acceleration(self, name: str, local_frame=True):
        raise NotImplementedError

    def get_body_contact_force(self, name: str):
        raise NotImplementedError
