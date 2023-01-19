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

    def get_com_pos(self):
        raise NotImplementedError

    def set_com_pos(self, pos: np.ndarray):
        raise NotImplementedError

    def get_torque(self):
        raise NotImplementedError

    def set_torque(self, torque: np.ndarray):
        raise NotImplementedError

    def sim_forward(self, dt: float = None):
        raise NotImplementedError

    def viewer_init(self):
        raise NotImplementedError

    def viewer_draw(self):
        raise NotImplementedError