class MjCassieSim(GenericSim):

    # @jeremy
    """
    A python wrapper around Mujoco python pkg that works better with Cassie???
    """

    def __init__(self) -> None:
        super().__init__()
        self.joint_inds = [4, 5,6]
        self.com_inds = [0, 1, 2]
        

    def get_joint_pops():
        return self.mj.qpos