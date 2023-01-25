import numpy as np

class GenericSim:
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

    def set_PD(self, P_targ: np.ndarray, D_targ: np.ndarray, P_gain: np.ndarray, D_gain: np.ndarray):
        raise NotImplementedError

    def hold(self):
        raise NotImplementedError

    def release(self):
        raise NotImplementedError
    
    def get_motor_pos(self):
        raise NotImplementedError

    def set_motor_pos(self, pos: np.ndarray):
        raise NotImplementedError

    def get_motor_vel(self):
        raise NotImplementedError

    def set_motor_vel(self, vel: np.ndarray):
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

    def set_base_position(self, pose: np.ndarray):
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
