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
        self.num_joint = 4
        # self.sim = CassieSim(modelfile=kwargs['modelfile'], terrain=kwargs['terrain'], perception=kwargs['perception'])
        # self.sim = CassieSim(terrain=kwargs['terrain'], perception=kwargs['perception'])
        self.sim = CassieSim()
        self.vis = None

        self.motor_pos_inds = [7, 8, 9, 14, 20, 21, 22, 23, 28, 34]
        self.joint_pos_inds = [15, 16, 29, 30]
        self.motor_vel_inds = [6, 7, 8, 12, 18, 19, 20, 21, 25, 31]
        self.joint_vel_inds = [13, 14, 26, 27]

        self.base_pos_inds = [0, 1, 2]
        self.base_orient_inds = [3, 4, 5, 6]
        self.base_trans_vel_inds = [0, 1, 2]
        self.base_rot_vel_inds = [3, 4, 5]

        self.P            = np.array([100,  100,  88,  96,  50, 100,  100,  88,  96,  50])
        self.D            = np.array([10.0, 10.0, 8.0, 9.6, 5.0, 10.0, 10.0, 8.0, 9.6, 5.0])
        self.offset       = np.array([0.0045, 0.0, 0.4973, -1.1997, -1.5968, 0.0045, 0.0, 0.4973, -1.1997, -1.5968])
        self.joint_limits_high = np.array([ 0.24, 0.25,  1.35, -0.82, -0.68, 0.2,   0.25,  1.35, -0.82, -0.68])
        self.joint_limits_low  = np.array([-0.2, -0.25, -0.8,  -2.0,  -2.0,  -0.24, -0.25, -0.8,  -2.0,  -2.0])
        self.u            = pd_in_t()
        self.robot_state = state_out_t()

    def get_joint_pos(self):
        return np.array(self.sim.qpos())[self.joint_pos_inds]

    def set_joint_pos(self, pos: np.ndarray):
        assert pos.ndim == 1, \
               f"set_joint_pos did not receive a 1 dimensional array"
        assert len(pos) == len(self.joint_pos_inds), \
               f"set_joint_pos did not receive array of size ({len(self.joint_pos_inds)})"
        curr_qpos = np.array(self.sim.qpos())
        curr_qpos[self.joint_pos_inds] = pos
        self.sim.set_qpos(curr_qpos)

    def get_joint_vel(self):
        return np.array(self.sim.qvel())[self.joint_vel_inds]

    def set_joint_vel(self, vel: np.ndarray):
        assert vel.ndim == 1, \
               f"set_joint_vel did not receive a 1 dimensional array"
        assert len(vel) == len(self.joint_vel_inds), \
               f"set_joint_vel did not receive array of size ({len(self.joint_vel_inds)})"
        curr_qvel = np.array(self.sim.qvel())
        curr_qvel[self.joint_vel_inds] = vel
        self.sim.set_qvel(curr_qvel)

    def get_motor_pos(self):
        return np.array(self.sim.qpos())[self.motor_pos_inds]

    def set_motor_pos(self, pos: np.ndarray):
        assert pos.ndim == 1, \
               f"set_motor_pos did not receive a 1 dimensional array"
        assert len(pos) == len(self.motor_pos_inds), \
               f"set_motor_pos did not receive array of size {len(self.motor_pos_inds)}"
        curr_qpos = np.array(self.sim.qpos())
        curr_qpos[self.motor_pos_inds] = pos
        self.sim.set_qpos(curr_qpos)

    def get_motor_vel(self):
        return np.array(self.sim.qvel())[self.motor_vel_inds]

    def set_motor_vel(self, vel: np.ndarray):
        assert vel.ndim == 1, \
               f"set_motor_vel did not receive a 1 dimensional array"
        assert len(vel) == len(self.motor_vel_inds), \
               f"set_motor_vel did not receive array of size {len(self.motor_vel_inds)}"
        curr_qvel = np.array(self.sim.qvel())
        curr_qvel[self.motor_vel_inds] = vel
        self.sim.set_qvel(curr_qvel)

    def get_com_pos(self):
        return np.array(self.sim.qpos())[self.base_pos_inds]

    def set_com_pos(self, pos: np.ndarray):
        assert pos.ndim == 1, \
               f"set_com_pos did not receive a 1 dimensional array"
        assert len(pos) == len(self.base_pos_inds), \
               f"set_com_pos did not receive array of size {len(self.base_pos_inds)}"
        curr_qpos = np.array(self.sim.qpos())
        curr_qpos[self.base_pos_inds] = pos
        self.sim.set_qpos(curr_qpos)

    def get_com_trans_vel(self):
        return np.array(self.sim.qvel())[self.base_trans_vel_inds]

    def set_com_trans_vel(self, vel: np.ndarray):
        assert vel.ndim == 1, \
               f"set_com_trans_vel did not receive a 1 dimensional array"
        assert len(vel) == len(self.base_trans_vel_inds), \
               f"set_com_trans_vel did not receive array of size {len(self.base_trans_vel_inds)}"
        curr_qvel = np.array(self.sim.qvel())
        curr_qvel[self.base_trans_vel_inds] = vel
        self.sim.set_qvel(curr_qvel)

    def get_com_quat(self):
        return np.array(self.sim.qvel())[self.base_orient_inds]

    def set_com_quat(self, quat: np.ndarray):
        assert quat.ndim == 1, \
               f"set_com_quat did not receive a 1 dimensional array"
        assert len(quat) == len(self.base_orient_inds), \
               f"set_com_quat did not receive array of size {len(self.base_orient_inds)}"
        curr_qpos = np.array(self.sim.qpos())
        curr_qpos[self.base_orient_inds] = quat
        self.sim.set_qpos(curr_qpos)

    def get_com_rot_vel(self):
        return np.array(self.sim.qvel())[self.base_rot_vel_inds]

    def set_com_rot_vel(self, vel: np.ndarray):
        assert vel.ndim == 1, \
               f"set_com_rot_vel did not receive a 1 dimensional array"
        assert len(vel) == len(self.base_rot_vel_inds), \
               f"set_com_rot_vel did not receive array of size {len(self.base_rot_vel_inds)}"
        curr_qvel = np.array(self.sim.qvel())
        curr_qvel[self.base_rot_vel_inds] = vel
        self.sim.set_qvel(curr_qvel)

    def get_torque(self):
        # TODO: Probably have to expose mjData.ctrl in libcassiemujoco
        pass

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

    def set_PD(self, P_targ: np.ndarray, D_targ: np.ndarray, P_gain: np.ndarray, D_gain: np.ndarray):
        assert P_targ.ndim == 1, \
               f"set_PD P_targ was not a 1 dimensional array"
        assert D_targ.ndim == 1, \
               f"set_PD D_targ was not a 1 dimensional array"
        assert P_gain.ndim == 1, \
               f"set_PD P_gain was not a 1 dimensional array"
        assert D_gain.ndim == 1, \
               f"set_PD D_gain was not a 1 dimensional array"
        assert len(P_targ) == self.num_actuators, \
               f"set_PD P_targ was not array of size {self.num_actuators}"
        assert len(D_targ) == self.num_actuators, \
               f"set_PD D_targ was not array of size {self.num_actuators}"
        assert len(P_gain) == self.num_actuators, \
               f"set_PD P_gain was not array of size {self.num_actuators}"
        assert len(D_gain) == self.num_actuators, \
               f"set_PD D_gain was not array of size {self.num_actuators}"
        self.u = pd_in_t()
        for i in range(5):
            self.u.leftLeg.motorPd.pGain[i]  = P_gain[i]
            self.u.rightLeg.motorPd.pGain[i] = P_gain[i + 5]

            self.u.leftLeg.motorPd.dGain[i]  = D_gain[i]
            self.u.rightLeg.motorPd.dGain[i] = D_gain[i + 5]

            self.u.leftLeg.motorPd.torque[i]  = 0  # Feedforward torque
            self.u.rightLeg.motorPd.torque[i] = 0

            self.u.leftLeg.motorPd.pTarget[i]  = P_targ[i]
            self.u.rightLeg.motorPd.pTarget[i] = P_targ[i + 5]

            self.u.leftLeg.motorPd.dTarget[i]  = D_targ[i]
            self.u.rightLeg.motorPd.dTarget[i] = D_targ[i + 5]

    def sim_forward(self, dt: float = None):
        # NOTE: Ok to assume libcassie always at 2kHz?
        model_dt = 0.0005
        if dt:
            num_steps = int(dt / model_dt)
            WARNING = '\033[93m'
            ENDC = '\033[0m'
            if num_steps * model_dt != dt:
                print(f"{WARNING}Warning: {dt} does not fit evenly within the sim timestep of"
                    f" {model_dt}, simulating forward"
                    f" {num_steps * model_dt}s instead.{ENDC}")
        else:
            num_steps = 1
        for i in range(num_steps):
            self.robot_state = self.sim.step_pd(self.u)

    def hold(self):
        self.sim.hold()

    def release(self):
        self.sim.release()

    def viewer_init(self):
        self.viewer = CassieVis(self.sim)

    def viewer_render(self):
        assert not self.viewer is None, \
               f"viewer has not been initalized yet, can not render"
        return self.viewer.draw(self.sim)

    def viewer_paused(self):
        assert not self.viewer is None, \
               f"viewer has not been initalized yet, can not check paused status"
        return self.viewer.ispaused()
