import pathlib

import mujoco as mj
import numpy as np
from ..MujocoViewer import MujocoViewer
from ..GenericSim import GenericSim

class MjCassieSim(GenericSim):

    # @jeremy
    """
    A python wrapper around Mujoco python pkg that works better with Cassie???
    """

    def __init__(self) -> None:
        super().__init__()
        model_path = pathlib.Path(__file__).parent.resolve() / "cassiemujoco/cassie.xml"
        self.model = mj.MjModel.from_xml_path(str(model_path))
        self.data = mj.MjData(self.model)
        self.viewer = None
        self.motor_inds = [7, 8, 9, 14, 20, 21, 22, 23, 28, 34]
        self.joint_inds = [15, 16, 29, 30]
        self.com_inds = [0, 1, 2]
        self.com_orient_inds = [3, 4, 5, 6]
        self.offset = np.array([0.0045, 0.0, 0.4973, -1.1997, -1.5968, 0.0045, 0.0, 0.4973, -1.1997, -1.5968])
        self.reset_qpos = np.array([0, 0, 1.01, 1, 0, 0, 0,
                    0.0045, 0, 0.4973, 0.9785, -0.0164, 0.01787, -0.2049,
                    -1.1997, 0, 1.4267, 0, -1.5244, 1.5244, -1.5968,
                    -0.0045, 0, 0.4973, 0.9786, 0.00386, -0.01524, -0.2051,
                    -1.1997, 0, 1.4267, 0, -1.5244, 1.5244, -1.5968])
        self.data.qpos = self.reset_qpos
        mj.mj_forward(self.model, self.data)

    def get_joint_pos(self):
        return self.data.qpos[self.joint_inds]

    def get_motor_pos(self):
        return self.data.qpos[self.motor_inds]

    def set_torque(self, torque):
        self.data.ctrl[:] = torque

    def step(self):
        mj.mj_step(self.model, self.data)

    def viewer_init(self):
        self.viewer = MujocoViewer(self.model, self.data, self.reset_qpos)

    def viewer_render(self):
        if self.viewer.is_alive:
            self.viewer.render()
        else:
            print("Error: Viewer not alive, can not render.")
            return
