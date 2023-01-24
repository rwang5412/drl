import numpy as np

class GenericSim:
    """
    A base class to define the functions that interact with simulator.
    This class contains a set of getter/setter that unify generic naming conventions for different simulators
    """

    def __init__(self) -> None:
        pass

    def get_joint_pos(self):
        raise NotImplementedError

    def set_joint_pos(self, pos: np.ndarray):
        raise NotImplementedError

    def get_joint_vel(self):
        raise NotImplementedError

    def set_joint_vel(self, vel: np.ndarray):
        raise NotImplementedError
    
    def get_motor_pos(self):
        raise NotImplementedError

    def set_motor_pos(self, pos: np.ndarray):
        raise NotImplementedError

    def get_motor_vel(self):
        raise NotImplementedError

    def set_motor_vel(self, vel: np.ndarray):
        raise NotImplementedError

    def get_com_pos(self):
        raise NotImplementedError

    def set_com_pos(self, pos: np.ndarray):
        raise NotImplementedError

    def get_com_trans_vel(self):
        raise NotImplementedError

    def set_com_trans_vel(self, vel: np.ndarray):
        raise NotImplementedError

    def get_com_quat(self):
        raise NotImplementedError

    def set_com_quat(self, quat: np.ndarray):
        raise NotImplementedError

    def get_com_rot_vel(self):
        raise NotImplementedError

    def set_com_rot_vel(self, vel: np.ndarray):
        raise NotImplementedError

    def get_torque(self):
        raise NotImplementedError

    def set_torque(self, torque: np.ndarray):
        raise NotImplementedError

    def set_PD(self, P_targ: np.ndarray, D_targ: np.ndarray, P_gain: np.ndarray, D_gain: np.ndarray):
        raise NotImplementedError()

    def sim_forward(self, dt: float = None):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def hold(self):
        raise NotImplementedError

    def release(self):
        raise NotImplementedError()

    def viewer_init(self):
        raise NotImplementedError

    def viewer_render(self):
        raise NotImplementedError