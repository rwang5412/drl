import mujoco as mj
import numpy as np
import pathlib

from sim import GenericSim, MujocoViewer

WARNING = '\033[93m'
ENDC = '\033[0m'

class MjCassieSim(GenericSim):

    """
    A python wrapper around Mujoco python pkg that works better with Cassie???
    """

    def __init__(self):
        super().__init__()
        model_path = pathlib.Path(__file__).parent.resolve() / "cassiemujoco/cassie.xml"
        self.model = mj.MjModel.from_xml_path(str(model_path))
        self.data = mj.MjData(self.model)
        self.viewer = None
        self.motor_position_inds = [7, 8, 9, 14, 20, 21, 22, 23, 28, 34]
        self.joint_position_inds = [15, 16, 29, 30]
        self.motor_velocity_inds = [6, 7, 8, 12, 18, 19, 20, 21, 25, 31]
        self.joint_velocity_inds = [13, 14, 26, 27]

        self.base_position_inds = [0, 1, 2]
        self.base_orientation_inds = [3, 4, 5, 6]
        self.base_linear_velocity_inds = [0, 1, 2]
        self.base_angular_velocity_inds = [3, 4, 5]
        self.base_body_name = "cassie-pelvis"
        self.feet_body_name = ["left-foot", "right-foot"]

        self.simulator_rate = int(1 / self.model.opt.timestep)
        self.num_actuators = 10
        self.num_joints = 4
        self.offset = np.array([0.0045, 0.0, 0.4973, -1.1997, -1.5968, 0.0045, 0.0, 0.4973, -1.1997, -1.5968])
        self.kp = np.array([100,  100,  88,  96,  50, 100, 100,  88,  96,  50])
        self.kd = np.array([10.0, 10.0, 8.0, 9.6, 5.0, 10.0, 10.0, 8.0, 9.6, 5.0])
        self.reset_qpos = np.array([0, 0, 1.01, 1, 0, 0, 0,
                    0.0045, 0, 0.4973, 0.9785, -0.0164, 0.01787, -0.2049,
                    -1.1997, 0, 1.4267, 0, -1.5244, 1.5244, -1.5968,
                    -0.0045, 0, 0.4973, 0.9786, 0.00386, -0.01524, -0.2051,
                    -1.1997, 0, 1.4267, 0, -1.5244, 1.5244, -1.5968])

    def reset(self, qpos: np.ndarray=None):
        if qpos:
            assert len(qpos) == self.model.nq, f"reset qpos len={len(qpos)}, but should be {self.model.nq}"
            self.data.qpos = qpos
        else:
            self.data.qpos = self.reset_qpos
        mj.mj_forward(self.model, self.data)

    def sim_forward(self, dt: float = None):
        if dt:
            num_steps = int(dt / self.model.opt.timestep)
            if num_steps * self.model.opt.timestep != dt:
                raise RuntimeError(f"{WARNING}Warning: {dt} does not fit evenly within the sim timestep of"
                    f" {self.model.opt.timestep}, simulating forward"
                    f" {num_steps * self.model.opt.timestep}s instead.{ENDC}")
        else:
            num_steps = 1
        mj.mj_step(self.model, self.data, nstep=num_steps)

    def set_torque(self, torque: np.ndarray):
        assert torque.shape == (self.num_actuators,), \
               f"set_torque got array of shape {torque.shape} but " \
               f"should be shape ({self.num_actuators},)."
        self.data.ctrl[:] = torque

    def set_PD(self,
               setpoint: np.ndarray,
               velocity: np.ndarray,
               kp: np.ndarray,
               kd: np.ndarray):
        args = locals() # This has to be the first line in the function
        for arg in args:
            if arg != "self":
                assert args[arg].shape == (self.model.nu,), \
                f"set_PD {arg} was not a 1 dimensional array of size {self.model.nu}"
        torque = kp * (setpoint - self.data.qpos[self.motor_position_inds]) + \
                 kd * (velocity - self.data.qvel[self.motor_velocity_inds])
        self.data.ctrl[:] = torque

    def hold(self):
        # Set stiffness/damping for body translation joints
        for i in range(3):
            self.model.jnt_stiffness[i] = 1e5
            self.model.dof_damping[i] = 1e4
            self.model.qpos_spring[i] = self.data.qpos[i]

        # Set damping for body rotation joint
        for i in range(3, 6):
            self.model.dof_damping[i] = 1e4

    def release(self):
        # Zero stiffness/damping for body translation joints
        for i in range(3):
            self.model.jnt_stiffness[i] = 0
            self.model.dof_damping[i] = 0

        # Zero damping for body rotation joint
        for i in range(3, 6):
            self.model.dof_damping[i] = 0

    def viewer_init(self, width=None, height=None, camera_id=-1):
        self.viewer = MujocoViewer(self.model, self.data, self.reset_qpos, width=width, \
            height=height, camera_id=camera_id)

    def viewer_render(self):
        assert not self.viewer is None, \
               f"viewer has not been initalized yet, can not render"
        if self.viewer.is_alive:
            return self.viewer.render()
        else:
            raise RuntimeError("Error: Viewer not alive, can not render. Check that viewer has not \
                  been destroyed.")

    def viewer_paused(self):
        assert not self.viewer is None, \
               f"viewer has not been initalized yet, can not check paused status"
        if self.viewer.is_alive:
            return self.viewer.paused
        else:
            raise RuntimeError("Error: Viewer not alive, can not check paused status. Check that \
                  viewer has not been destroyed.")

    """The followings are getter/setter functions to unify with naming with GenericSim()
    """
    def get_joint_position(self):
        return self.data.qpos[self.joint_position_inds]

    def get_joint_velocity(self):
        return self.data.qvel[self.joint_velocity_inds]

    def get_motor_position(self):
        return self.data.qpos[self.motor_position_inds]

    def get_motor_velocity(self):
        return self.data.qvel[self.motor_velocity_inds]

    def get_base_position(self):
        return self.data.qpos[self.base_position_inds]

    def get_base_linear_velocity(self):
        return self.data.qvel[self.base_linear_velocity_inds]

    def get_base_orientation(self):
        return self.data.qpos[self.base_orientation_inds]

    def get_base_angular_velocity(self):
        return self.data.qvel[self.base_angular_velocity_inds]

    def get_torque(self):
        return self.data.ctrl[:]

    def get_joint_qpos_adr(self, name: str):
        return self.model.jnt_qposadr[mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, name)]

    def get_joint_dof_adr(self, name: str):
        return self.model.jnt_dofadr[mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, name)]

    def get_simulation_time(self):
        return self.data.time

    def get_body_pose(self, name: str):
        """Get body pose by name

        Args:
            name (str): body name

        Returns:
            ndarray: pose [3xlinear, 4xquaternion]
        """
        body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, name)
        pose = np.zeros(7)
        pose[:3] = self.data.xpos[body_id]
        pose[3:] = self.data.xquat[body_id]
        return pose

    def get_body_velocity(self, name: str, local_frame=False):
        """Get body velocity by name

        Args:
            name (str): body name
            local_frame (bool, optional): Defaults to False.

        Returns:
            ndarray: velocity [3xlinear, 3xangular]
        """
        body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, name)
        velocity = np.zeros(6)
        mj.mj_objectVelocity(self.model, self.data, mj.mjtObj.mjOBJ_BODY, body_id, velocity, local_frame)
        tmp = velocity[3:6].copy()
        velocity[3:6] = velocity[0:3]
        velocity[0:3] = tmp
        return velocity

    def get_body_contact_force(self, name: str):
        """Get contact forces for named body in contact frame

        Args:
            name (str): body name

        Returns:
            ndarray: sum of all wrenches acting on the body
        """
        body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, name)
        # Sum over all contact wrenches over possible geoms within the body
        total_wrench = np.zeros(6)
        contact_points = 0
        for contact_id in range(self.data.ncon):
            if body_id == self.model.geom_bodyid[self.data.contact[contact_id].geom1] or \
               body_id == self.model.geom_bodyid[self.data.contact[contact_id].geom2]:
                   contact_wrench_point = np.zeros(6)
                   mj.mj_contactForce(self.model, self.data, contact_id, contact_wrench_point)
                   total_wrench += contact_wrench_point
                   contact_points += 1
        return total_wrench

    def set_joint_position(self, position: np.ndarray):
        assert position.shape == (self.num_joints,), \
               f"set_joint_position got array of shape {position.shape} but " \
               f"should be shape ({self.num_joints},)."
        self.data.qpos[self.joint_position_inds] = position
        mj.mj_forward(self.model, self.data)

    def set_joint_velocity(self, velocity: np.ndarray):
        assert velocity.shape == (self.num_joints,), \
               f"set_joint_velocity got array of shape {velocity.shape} but " \
               f"should be shape ({self.num_joints},)."
        self.data.qvel[self.joint_velocity_inds] = velocity
        mj.mj_forward(self.model, self.data)

    def set_motor_position(self, position: np.ndarray):
        assert position.shape == (self.num_actuators,), \
               f"set_motor_position got array of shape {position.shape} but " \
               f"should be shape ({self.num_actuators},)."
        self.data.qpos[self.motor_position_inds] = position
        mj.mj_forward(self.model, self.data)

    def set_motor_velocity(self, velocity: np.ndarray):
        assert velocity.shape == (self.num_actuators,), \
               f"set_motor_velocity got array of shape {velocity.shape} but " \
               f"should be shape ({self.num_actuators},)."
        self.data.qvel[self.motor_velocity_inds] = velocity
        mj.mj_forward(self.model, self.data)

    def set_base_position(self, position: np.ndarray):
        assert position.shape == (3,), \
               f"set_base_position got array of shape {position.shape} but " \
               f"should be shape (3,)."
        self.data.qpos[self.base_position_inds] = position
        mj.mj_forward(self.model, self.data)

    def set_base_linear_velocity(self, velocity: np.ndarray):
        assert velocity.shape == (3,), \
               f"set_base_linear_velocity got array of shape {velocity.shape} but " \
               f"should be shape (3,)."
        self.data.qvel[self.base_linear_velocity_inds] = velocity
        mj.mj_forward(self.model, self.data)

    def set_base_orientation(self, quat: np.ndarray):
        assert quat.shape == (4,), \
               f"set_base_orientation got array of shape {quat.shape} but " \
               f"should be shape (4,)."
        self.data.qpos[self.base_orientation_inds] = quat
        mj.mj_forward(self.model, self.data)

    def set_base_angular_velocity(self, velocity: np.ndarray):
        assert velocity.shape == (3,), \
               f"set_base_angular_velocity got array of shape {velocity.shape} but " \
               f"should be shape (3,)."
        self.data.qvel[self.base_angular_velocity_inds] = velocity
        mj.mj_forward(self.model, self.data)

