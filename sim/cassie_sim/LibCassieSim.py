import numpy as np
import pathlib
import time

from .cassiemujoco import pd_in_t, state_out_t, CassieSim, CassieVis
from ..GenericSim import GenericSim


class LibCassieSim(GenericSim):

    # @jeremy
    """
    Cassie simulation using Agility compiled C library libcassiemujoco.so. Uses Mujoco under the
    hood, simulation code is contained in `cassiemujoco` folder.
    """

    def __init__(self, *args, **kwargs) -> None:
        self.state_est_size  = 35
        self.num_actuators   = 10
        # self.sim = CassieSim(modelfile=kwargs['modelfile'], terrain=kwargs['terrain'], perception=kwargs['perception'])
        # self.sim = CassieSim(terrain=kwargs['terrain'], perception=kwargs['perception'])
        self.sim = CassieSim()
        self.vis = None

        self.motor_pos_idx      = [7, 8, 9, 14, 20, 21, 22, 23, 28, 34]
        self.motor_vel_idx      = [6, 7, 8, 12, 18, 19, 20, 21, 25, 31]
        self.joint_inds = [15, 16, 29, 30]
        self.P            = np.array([100,  100,  88,  96,  50])
        self.D            = np.array([10.0, 10.0, 8.0, 9.6, 5.0])
        self.offset       = np.array([0.0045, 0.0, 0.4973, -1.1997, -1.5968, 0.0045, 0.0, 0.4973, -1.1997, -1.5968])
        self.joint_limits_high = np.array([ 0.24, 0.25,  1.35, -0.82, -0.68, 0.2,   0.25,  1.35, -0.82, -0.68])
        self.joint_limits_low  = np.array([-0.2, -0.25, -0.8,  -2.0,  -2.0,  -0.24, -0.25, -0.8,  -2.0,  -2.0])
        self.u            = pd_in_t()
        self.robot_state = state_out_t()

    def get_joint_pos(self):
        return self.sim.qpos()[self.joint_inds]

    def get_motor_pos(self):
        return self.sim.qpos()[self.motor_pos_inds]

    def set_torque(self, torque: np.ndarray):
        # Only setting self.u, not actually calling step yet
        # Assume that torque order follows qpos order, so left leg and then right leg
        self.u = pd_in_t()
        for i in range(5):
            self.u.leftLeg.motorPd.pGain[i]  = 0
            self.u.rightLeg.motorPd.pGain[i] = 0

            self.u.leftLeg.motorPd.dGain[i]  = 0
            self.u.rightLeg.motorPd.dGain[i] = 0

            self.u.leftLeg.motorPd.torque[i]  = torque[i]  # Feedforward torque
            self.u.rightLeg.motorPd.torque[i] = torque[i+5]

            self.u.leftLeg.motorPd.pTarget[i]  = 0
            self.u.rightLeg.motorPd.pTarget[i] = 0

            self.u.leftLeg.motorPd.dTarget[i]  = 0
            self.u.rightLeg.motorPd.dTarget[i] = 0

    def sim_forward(self, dt: float = None):
        # NOTE: Ok to assume libcassie always at 2kHz?
        if dt:
            num_step = dt // 0.0005
        else:
            num_step = 1
        for i in range(num_step):
            self.robot_state = self.sim.step_pd(self.u)

    def viewer_init(self):
        self.viewer = CassieVis(self.sim)

    def viewer_render(self):
        self.viewer.draw(self.sim)
