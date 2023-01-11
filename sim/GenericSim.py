class GenericSim:
    """
    A base class to define the functions that interact with simulator.
    This class contains a set of getter/setter that unify generic naming conventions for different simulators
    """

    def __init__(self) -> None:
        pass
    
    def get_joint_pos():
        raise NotImplementedError

    def set_joint_pos():
        raise NotImplementedError
    
    def get_joint_vel():
        raise NotImplementedError
    
    def set_joint_vel():
        raise NotImplementedError

    def get_com_pos():
        raise NotImplementedError
    
    def set_com_pos():
        raise NotImplementedError

    def step():
        raise NotImplementedError