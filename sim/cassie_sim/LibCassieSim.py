import numpy as np
import pathlib
import time

from .cassiemujoco import pd_in_t, state_out_t, CassieSim, CassieVis
from ..GenericSim import GenericSim


class LibCassieSim(GenericSim):

    """
    Cassie simulation using Agility compiled C library libcassiemujoco.so. Uses Mujoco under the
    hood, simulation code is contained in `cassiemujoco` folder.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.state_est_size  = 35
        self.num_actuators   = 10
        self.num_joints = 4
        # self.sim = CassieSim(modelfile=kwargs['modelfile'], terrain=kwargs['terrain'], perception=kwargs['perception'])
        # self.sim = CassieSim(terrain=kwargs['terrain'], perception=kwargs['perception'])
        self.sim = CassieSim()
        self.viewer = None
        self.sim_dt = 0.0005    # Assume libcassie sim is always 2kHz

        self.motor_position_inds = [7, 8, 9, 14, 20, 21, 22, 23, 28, 34]
        self.joint_position_inds = [15, 16, 29, 30]
        self.motor_velocity_inds = [6, 7, 8, 12, 18, 19, 20, 21, 25, 31]
        self.joint_velocity_inds = [13, 14, 26, 27]

        self.base_position_inds = [0, 1, 2]
        self.base_orientation_inds = [3, 4, 5, 6]
        self.base_linear_velocity_inds = [0, 1, 2]
        self.base_angular_velocity_inds = [3, 4, 5]

        self.num_actuators = 10
        self.num_joints = 4
        self.kp            = np.array([100,  100,  88,  96,  50, 100,  100,  88,  96,  50])
        self.kd            = np.array([10.0, 10.0, 8.0, 9.6, 5.0, 10.0, 10.0, 8.0, 9.6, 5.0])
        self.offset        = np.array([0.0045, 0.0, 0.4973, -1.1997, -1.5968, 0.0045, 0.0, 0.4973, -1.1997, -1.5968])
        self.joint_limits_high = np.array([ 0.24, 0.25,  1.35, -0.82, -0.68, 0.2,   0.25,  1.35, -0.82, -0.68])
        self.joint_limits_low  = np.array([-0.2, -0.25, -0.8,  -2.0,  -2.0,  -0.24, -0.25, -0.8,  -2.0,  -2.0])
        self.u            = pd_in_t()
        self.robot_state = state_out_t()

        self.reset_qpos = np.array([0, 0, 1.01, 1, 0, 0, 0,
                    0.0045, 0, 0.4973, 0.9785, -0.0164, 0.01787, -0.2049,
                    -1.1997, 0, 1.4267, 0, -1.5244, 1.5244, -1.5968,
                    -0.0045, 0, 0.4973, 0.9786, 0.00386, -0.01524, -0.2051,
                    -1.1997, 0, 1.4267, 0, -1.5244, 1.5244, -1.5968])

    def reset(self, qpos: np.ndarray=None):
        self.sim.set_const()
        if qpos:
            assert len(qpos) == self.model.nq, f"reset qpos len={len(qpos)}, but should be {self.model.nq}"
            self.sim.set_qpos(qpos)
        else:
            self.sim.set_qpos(self.reset_qpos)

    def sim_forward(self, dt: float = None):
        if dt:
            num_steps = int(dt / self.sim_dt)
            WARNING = '\033[93m'
            ENDC = '\033[0m'
            if num_steps * self.sim_dt != dt:
                print(f"{WARNING}Warning: {dt} does not fit evenly within the sim timestep of"
                    f" {self.sim_dt}, simulating forward"
                    f" {num_steps * self.sim_dt}s instead.{ENDC}")
        else:
            num_steps = 1
        for i in range(num_steps):
            self.robot_state = self.sim.step_pd(self.u)

    def set_torque(self, torque: np.ndarray):
        assert torque.shape == (self.num_actuators,), \
               f"set_torque got array of shape {torque.shape} but " \
               f"should be shape ({self.num_actuators},)."
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

    def set_PD(self,
               setpoint: np.ndarray,
               velocity: np.ndarray,
               kp: np.ndarray,
               kd: np.ndarray):
        args = locals() # This has to be the first line in the function
        for arg in args:
            if arg != "self":
                assert args[arg].shape == (self.num_actuators,), \
                f"set_PD {arg} was not a 1 dimensional array of size {self.num_actuators}"
        self.u = pd_in_t()
        for i in range(5):
            self.u.leftLeg.motorPd.pGain[i]  = kp[i]
            self.u.rightLeg.motorPd.pGain[i] = kp[i + 5]

            self.u.leftLeg.motorPd.dGain[i]  = kd[i]
            self.u.rightLeg.motorPd.dGain[i] = kd[i + 5]

            self.u.leftLeg.motorPd.torque[i]  = 0  # Feedforward torque
            self.u.rightLeg.motorPd.torque[i] = 0

            self.u.leftLeg.motorPd.pTarget[i]  = setpoint[i]
            self.u.rightLeg.motorPd.pTarget[i] = setpoint[i + 5]

            self.u.leftLeg.motorPd.dTarget[i]  = velocity[i]
            self.u.rightLeg.motorPd.dTarget[i] = velocity[i + 5]

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

    """
    The followings are getter/setter functions to unify with naming with GenericSim()
    """

    def get_joint_position(self):
        return np.array(self.sim.qpos())[self.joint_position_inds]

    def get_joint_velocity(self):
        return np.array(self.sim.qvel())[self.joint_velocity_inds]

    def get_motor_position(self):
        return np.array(self.sim.qpos())[self.motor_position_inds]

    def get_motor_velocity(self):
        return np.array(self.sim.qvel())[self.motor_velocity_inds]

    def get_base_position(self):
        return np.array(self.sim.qpos())[self.base_position_inds]

    def get_base_linear_velocity(self):
        return np.array(self.sim.qvel())[self.base_linear_velocity_inds]

    def get_base_orientation(self):
        return np.array(self.sim.qvel())[self.base_orientation_inds]

    def get_base_angular_velocity(self):
        return np.array(self.sim.qvel())[self.base_angular_velocity_inds]

    def get_torque(self):
        # NOTE: This returns mjData.ctrl which may not actually reflect the current command if
        # sim_forward has not been called yet. For example, if run `set_torque(trq)` and then call
        # `get_torque()` right after, it will not return the same `trq` array. This is because
        # `set_torque` uses the pd_in_t struct, which doesn't actually write to mjData.ctrl until
        # sim_forward, i.e sim.step_pd, is called.
        return np.array(self.sim.ctrl())

    def get_joint_qpos_adr(self, name: str):
        jnt_ind = self.sim.mj_name2id("joint", name)
        return self.sim.jnt_qposadr()[self.sim.mj_name2id("joint", name)]

    def get_joint_dof_adr(self, name: str):
        return self.sim.jnt_dofadr()[self.sim.mj_name2id("joint", name)]

    def set_joint_position(self, position: np.ndarray):
        assert position.shape == (self.num_joints,), \
               f"set_joint_position got array of shape {position.shape} but " \
               f"should be shape ({self.num_joints},)."
        curr_qpos = np.array(self.sim.qpos())
        curr_qpos[self.joint_position_inds] = position
        self.sim.set_qpos(curr_qpos)

    def set_joint_velocity(self, velocity: np.ndarray):
        assert velocity.shape == (self.num_joints,), \
               f"set_joint_velocity got array of shape {velocity.shape} but " \
               f"should be shape ({self.num_joints},)."
        curr_qvel = np.array(self.sim.qvel())
        curr_qvel[self.joint_velocity_inds] = velocity
        self.sim.set_qvel(curr_qvel)

    def set_motor_position(self, position: np.ndarray):
        assert position.shape == (self.num_actuators,), \
               f"set_motor_position got array of shape {position.shape} but " \
               f"should be shape ({self.num_actuators},)."
        curr_qpos = np.array(self.sim.qpos())
        curr_qpos[self.motor_position_inds] = position
        self.sim.set_qpos(curr_qpos)

    def set_motor_velocity(self, velocity: np.ndarray):
        assert velocity.shape == (self.num_actuators,), \
               f"set_motor_velocity got array of shape {velocity.shape} but " \
               f"should be shape ({self.num_actuators},)."
        curr_qvel = np.array(self.sim.qvel())
        curr_qvel[self.motor_velocity_inds] = velocity
        self.sim.set_qvel(curr_qvel)

    def set_base_position(self, position: np.ndarray):
        assert position.shape == (3,), \
               f"set_base_position got array of shape {position.shape} but " \
               f"should be shape (3,)."
        curr_qpos = np.array(self.sim.qpos())
        curr_qpos[self.base_position_inds] = position
        self.sim.set_qpos(curr_qpos)

    def set_base_linear_velocity(self, velocity: np.ndarray):
        assert velocity.shape == (3,), \
               f"set_base_linear_velocity got array of shape {velocity.shape} but " \
               f"should be shape (3,)."
        curr_qvel = np.array(self.sim.qvel())
        curr_qvel[self.base_linear_velocity_inds] = velocity
        self.sim.set_qvel(curr_qvel)

    def set_base_orientation(self, quat: np.ndarray):
        assert quat.shape == (4,), \
               f"set_base_orientation got array of shape {quat.shape} but " \
               f"should be shape (4,)."
        curr_qpos = np.array(self.sim.qpos())
        curr_qpos[self.base_orientation_inds] = quat
        self.sim.set_qpos(curr_qpos)

    def set_base_angular_velocity(self, velocity: np.ndarray):
        assert velocity.shape == (3,), \
               f"set_base_angular_velocity got array of shape {velocity.shape} but " \
               f"should be shape (3,)."
        curr_qvel = np.array(self.sim.qvel())
        curr_qvel[self.base_angular_velocity_inds] = velocity
        self.sim.set_qvel(curr_qvel)
