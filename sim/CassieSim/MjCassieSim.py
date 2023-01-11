import pathlib

import mujoco as mj
import mujoco_viewer
from ..GenericSim import GenericSim

class MjCassieSim(GenericSim):

    # @jeremy
    """
    A python wrapper around Mujoco python pkg that works better with Cassie???
    """

    def __init__(self) -> None:
        super().__init__()
        model_path = pathlib.Path(__file__).parent.resolve() / "cassie.xml"
        self.model = mj.MjModel.from_xml_path(str(model_path))
        self.data = mj.MjData(self.model)
        self.viewer = None
        self.motor_inds = [7, 8, 9, 14, 20, 21, 22, 23, 28, 34]
        self.joint_inds = [15, 16, 29, 30]
        self.com_inds = [0, 1, 2]
        self.com_orient_inds = [3, 4, 5, 6]

    def get_joint_pos(self):
        return self.data.qpos[self.joint_inds]

    def get_motor_pos(self):
        return self.data.qpos[self.motor_inds]

    def set_torque(self, torque):
        self.data.ctrl[:] = torque

    def step(self):
        mj.mj_step(self.model, self.data)

    def viewer_init(self):
        self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)

    def viewer_render(self):
        if self.viewer.is_alive:
            self.viewer.render()
        else:
            print("Error: Viewer not alive, can not render.")
            return
