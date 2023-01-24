import mujoco as mj
import numpy as np
import pathlib

from ..GenericSim import GenericSim
from ..MujocoViewer import MujocoViewer

class MjCassieSim(GenericSim):

    # @jeremy
    """
    A python wrapper around Mujoco python pkg that works better with Cassie???
    """

    def __init__(self):
        super().__init__()
        model_path = pathlib.Path(__file__).parent.resolve() / "cassiemujoco/cassie.xml"
        self.model = mj.MjModel.from_xml_path(str(model_path))
        self.data = mj.MjData(self.model)
        self.viewer = None
        self.motor_pos_inds = [7, 8, 9, 14, 20, 21, 22, 23, 28, 34]
        self.joint_pos_inds = [15, 16, 29, 30]
        self.motor_vel_inds = [6, 7, 8, 12, 18, 19, 20, 21, 25, 31]
        self.joint_vel_inds = [13, 14, 26, 27]

        self.base_pos_inds = [0, 1, 2]
        self.base_orient_inds = [3, 4, 5, 6]
        self.base_trans_vel_inds = [0, 1, 2]
        self.base_rot_vel_inds = [3, 4, 5]

        self.num_actuators = 10
        self.num_joint = 4
        self.offset = np.array([0.0045, 0.0, 0.4973, -1.1997, -1.5968, 0.0045, 0.0, 0.4973, -1.1997, -1.5968])
        self.kp            = np.array([100,  100,  88,  96,  50, 100, 100,  88,  96,  50])
        self.kd            = np.array([10.0, 10.0, 8.0, 9.6, 5.0, 10.0, 10.0, 8.0, 9.6, 5.0])
        self.reset_qpos = np.array([0, 0, 1.01, 1, 0, 0, 0,
                    0.0045, 0, 0.4973, 0.9785, -0.0164, 0.01787, -0.2049,
                    -1.1997, 0, 1.4267, 0, -1.5244, 1.5244, -1.5968,
                    -0.0045, 0, 0.4973, 0.9786, 0.00386, -0.01524, -0.2051,
                    -1.1997, 0, 1.4267, 0, -1.5244, 1.5244, -1.5968])
        self.data.qpos = self.reset_qpos
        mj.mj_forward(self.model, self.data)

    def get_joint_pos(self):
        return self.data.qpos[self.joint_pos_inds]

    def set_joint_pos(self, pos: np.ndarray):
        assert pos.ndim == 1, \
               f"set_joint_pos did not receive a 1 dimensional array"
        assert len(pos) == len(self.joint_pos_inds), \
               f"set_joint_pos did not receive array of size ({len(self.joint_pos_inds)})"
        self.data.qpos[self.joint_pos_inds] = pos
        mj.mj_forward(self.model, self.data)

    def get_joint_vel(self):
        return self.data.qvel[self.joint_vel_inds]

    def set_joint_vel(self, vel: np.ndarray):
        assert vel.ndim == 1, \
               f"set_joint_vel did not receive a 1 dimensional array"
        assert len(vel) == len(self.joint_vel_inds), \
               f"set_joint_vel did not receive array of size ({len(self.joint_vel_inds)})"
        self.data.qvel[self.joint_vel_inds] = vel
        mj.mj_forward(self.model, self.data)

    def get_motor_pos(self):
        return self.data.qpos[self.motor_pos_inds]

    def set_motor_pos(self, pos: np.ndarray):
        assert pos.ndim == 1, \
               f"set_motor_pos did not receive a 1 dimensional array"
        assert len(pos) == len(self.motor_pos_inds), \
               f"set_motor_pos did not receive array of size {len(self.motor_pos_inds)}"
        self.data.qpos[self.motor_pos_inds] = pos
        mj.mj_forward(self.model, self.data)

    def get_motor_vel(self):
        return self.data.qvel[self.motor_vel_inds]

    def set_motor_vel(self, vel: np.ndarray):
        assert vel.ndim == 1, \
               f"set_motor_vel did not receive a 1 dimensional array"
        assert len(vel) == len(self.motor_vel_inds), \
               f"set_motor_vel did not receive array of size {len(self.motor_vel_inds)}"
        self.data.qvel[self.motor_vel_inds] = vel
        mj.mj_forward(self.model, self.data)

    def get_com_pos(self):
        return self.data.qpos[self.base_pos_inds]

    def set_com_pos(self, pos: np.ndarray):
        assert pos.ndim == 1, \
               f"set_com_pos did not receive a 1 dimensional array"
        assert len(pos) == len(self.base_pos_inds), \
               f"set_com_pos did not receive array of size {len(self.base_pos_inds)}"
        self.data.qpos[self.base_pos_inds] = pos
        mj.mj_forward(self.model, self.data)

    def get_com_trans_vel(self):
        return self.data.qvel[self.base_trans_vel_inds]

    def set_com_trans_vel(self, vel: np.ndarray):
        assert vel.ndim == 1, \
               f"set_com_trans_vel did not receive a 1 dimensional array"
        assert len(vel) == len(self.base_trans_vel_inds), \
               f"set_com_trans_vel did not receive array of size {len(self.base_trans_vel_inds)}"
        self.data.qvel[self.base_trans_vel_inds] = vel
        mj.mj_forward(self.model, self.data)

    def get_com_quat(self):
        return self.data.qpos[self.base_orient_inds]

    def set_com_quat(self, quat: np.ndarray):
        assert quat.ndim == 1, \
               f"set_com_quat did not receive a 1 dimensional array"
        assert len(quat) == len(self.base_orient_inds), \
               f"set_com_quat did not receive array of size {len(self.base_orient_inds)}"
        self.data.qpos[self.base_orient_inds] = quat
        mj.mj_forward(self.model, self.data)

    def get_com_rot_vel(self):
        return self.data.qpos[self.base_rot_vel_inds]

    def set_com_rot_vel(self, vel: np.ndarray):
        assert vel.ndim == 1, \
               f"set_com_rot_vel did not receive a 1 dimensional array"
        assert len(vel) == len(self.base_rot_vel_inds), \
               f"set_com_rot_vel did not receive array of size {len(self.base_rot_vel_inds)}"
        self.data.qvel[self.base_rot_vel_inds] = vel
        mj.mj_forward(self.model, self.data)

    def get_torque(self):
        return self.data.ctrl[:]

    def set_torque(self, torque: np.ndarray):
        assert torque.ndim == 1, \
               f"set_torque did not receive a 1 dimensional array"
        assert len(torque) == self.model.nu, \
               f"set_torque did not receive array of size {self.model.nu}"
        self.data.ctrl[:] = torque

    def set_PD(self, P_targ: np.ndarray, D_targ: np.ndarray, P_gain: np.ndarray, D_gain: np.ndarray):
        assert P_targ.ndim == 1, \
               f"set_PD P_targ was not a 1 dimensional array"
        assert D_targ.ndim == 1, \
               f"set_PD D_targ was not a 1 dimensional array"
        assert P_gain.ndim == 1, \
               f"set_PD P_gain was not a 1 dimensional array"
        assert D_gain.ndim == 1, \
               f"set_PD D_gain was not a 1 dimensional array"
        assert len(P_targ) == self.model.nu, \
               f"set_PD P_targ was not array of size {self.model.nu}"
        assert len(D_targ) == self.model.nu, \
               f"set_PD D_targ was not array of size {self.model.nu}"
        assert len(P_gain) == self.model.nu, \
               f"set_PD P_gain was not array of size {self.model.nu}"
        assert len(D_gain) == self.model.nu, \
               f"set_PD D_gain was not array of size {self.model.nu}"
        torque = P_gain * (P_targ - self.data.qpos[self.motor_pos_inds]) + \
                 D_gain * (D_targ - self.data.qvel[self.motor_vel_inds])
        self.data.ctrl[:] = torque

    def sim_forward(self, dt: float = None):
        if dt:
            num_steps = int(dt / self.model.opt.timestep)
            WARNING = '\033[93m'
            ENDC = '\033[0m'
            if num_steps * self.model.opt.timestep != dt:
                print(f"{WARNING}Warning: {dt} does not fit evenly within the sim timestep of"
                    f" {self.model.opt.timestep}, simulating forward"
                    f" {num_steps * self.model.opt.timestep}s instead.{ENDC}")
        else:
            num_steps = 1
        mj.mj_step(self.model, self.data, nstep=num_steps)

    def reset(self):
        mj.resetData(self.model, self.data)
        self.data.qpos = self.reset_qpos
        mj.mj_forward(self.model, self.data)

    def hold(self):
        # Set stiffness/damping for body translation joints
        # NOTE: Ok to assume first 6 joints are com joints?
        for i in range(3):
            self.model.jnt_stiffness[i] = 1e5
            self.model.dof_damping[i] = 1e4
            self.model.qpos_spring[i] = self.data.qpos[i]

        # Set damping for body rotation joint
        for i in range(3, 6):
            self.model.dof_damping[i] = 1e4

    def release(self):
        # Zero stiffness/damping for body translation joints
        # NOTE: Ok to assume first 6 joints are com joints?
        for i in range(3):
            self.model.jnt_stiffness[i] = 0
            self.model.dof_damping[i] = 0

        # Zero damping for body rotation joint
        for i in range(3, 6):
            self.model.dof_damping[i] = 0

    def viewer_init(self):
        self.viewer = MujocoViewer(self.model, self.data, self.reset_qpos)

    def viewer_render(self):
        assert not self.viewer is None, \
               f"viewer has not been initalized yet, can not render"
        if self.viewer.is_alive:
            self.viewer.render()
            return True
        else:
            print("Error: Viewer not alive, can not render.")
            return False

    def viewer_paused(self):
        assert not self.viewer is None, \
               f"viewer has not been initalized yet, can not check paused status"
        if self.viewer.is_alive:
            return self.viewer.paused
        else:
            print("Error: Viewer not alive, can not check paused status.")
            return False
