import copy
import numpy as np
import mujoco as mj

from .generic_sim import GenericSim
from .mujoco_viewer import MujocoViewer
from .mujoco_render import MujocoRender
from util.colors import FAIL, WARNING, ENDC
from sim.util.geom import Geom

class MujocoSim(GenericSim):
    """
    A base class to define general useful functions that interact with Mujoco simulator.
    This class explicitly avoids robot-specific names.
    """

    def __init__(self, model_path, terrain=None):
        super().__init__()
        self.model = mj.MjModel.from_xml_path(str(model_path))
        self.data = mj.MjData(self.model)
        self.nq = self.model.nq
        self.nv = self.model.nv
        self.nu = self.model.nu
        self.nbody = self.model.nbody
        self.njnt = self.model.njnt
        self.ngeom = self.model.ngeom
        self.viewer = None
        self.terrain = terrain
        # Enforce that necessary constants and index arrays are defined
        check_vars = ["torque_delay_cycles", "torque_efficiency", "motor_position_inds",
            "motor_velocity_inds", "joint_position_inds", "joint_velocity_inds",
            "base_position_inds", "base_orientation_inds", "base_linear_velocity_inds",
            "base_angular_velocity_inds", "num_actuators", "num_joints", "reset_qpos"]
        for var in check_vars:
            assert hasattr(self, var), \
                f"{FAIL}Env {self.__class__.__name__} has not defined self.{var}.{ENDC}"
            assert getattr(self, var) is not None, \
                f"{FAIL}In env {self.__class__.__name__} self.{var} is None.{ENDC}"

        assert self.torque_delay_cycles > 0, \
            f"{FAIL}Env {self.__class__.__name__} must have non-zeron torque_delay_cycles. Note " \
            f"that delay cycle of 1 corresponds to no delay (specifies size of the torque buffer.{ENDC}"
        self.torque_buffer = np.zeros((self.torque_delay_cycles, self.model.nu))

        # Load geoms/bodies for hfield/box/obstacle/stone/stair
        self.load_fixed_object()
        # self.load_movable_object()

    def load_fixed_object(self, num_geoms_in_xml=20):
        """Load any geoms. can add more types, such as non box types, but is limited at compile.
        """
        try:
            self.box_geoms = [f'box{i}' for i in range(num_geoms_in_xml)]
            self.geom_generator = Geom(self)
        except:
            print(f"No box-typed geom listed in XML.\n"
                  f"Or num of geoms is not equal to {num_geoms_in_xml}.")

    def reset(self, qpos: np.ndarray=None):
        mj.mj_resetData(self.model, self.data)
        if qpos is not None:
            assert len(qpos) == self.model.nq, \
                f"{FAIL}reset qpos len={len(qpos)}, but should be {self.model.nq}{ENDC}"
            self.data.qpos = qpos
        else:
            self.data.qpos = self.reset_qpos
        mj.mj_forward(self.model, self.data)

    def sim_forward(self, dt: float = None):
        if dt:
            num_steps = int(dt / self.model.opt.timestep)
            if num_steps * self.model.opt.timestep != dt:
                raise RuntimeError(f"{WARNING}Warning: {dt} does not fit evenly within the sim "
                    f"timestep of {self.model.opt.timestep}, simulating forward "
                    f"{num_steps * self.model.opt.timestep}s instead.{ENDC}")
        else:
            num_steps = 1
        mj.mj_step(self.model, self.data, nstep=num_steps)

    def set_torque(self, torque: np.ndarray):
        assert torque.shape == (self.num_actuators,), \
               f"{FAIL}set_torque got array of shape {torque.shape} but " \
               f"should be shape ({self.num_actuators},).{ENDC}"
        # Apply next torque command in buffer
        self.data.ctrl[:] = self.torque_buffer[0, :]
        # Shift torque buffer values and append new command at the end
        self.torque_buffer = np.roll(self.torque_buffer, -1, axis = 0)
        self.torque_buffer[-1, :] = self.torque_efficiency * torque / self.model.actuator_gear[:, 0]

    def set_PD(self,
               setpoint: np.ndarray,
               velocity: np.ndarray,
               kp: np.ndarray,
               kd: np.ndarray):
        args = locals() # This has to be the first line in the function
        for arg in args:
            if arg != "self":
                assert args[arg].shape == (self.model.nu,), \
                f"{FAIL}set_PD {arg} was not a 1 dimensional array of size {self.model.nu}{ENDC}"
        torque = kp * (setpoint - self.data.qpos[self.motor_position_inds])
        # Explicit damping
        torque += kd * velocity
        # Implicit damping
        self.model.dof_damping[self.motor_velocity_inds] = kd
        self.set_torque(torque)

    def hold(self):
        """Set stiffness/damping for base 6DOF so base is fixed
        NOTE: There is an old funky stuff when left hip-roll motor is somehow coupled with the base
        joint, so left-hip-roll is not doing things correctly when holding.
        Turns out xml seems need to be defined with 3 slide and 1 ball instead of free joint.
        """
        for i in range(3):
            self.model.jnt_stiffness[i] = 1e5
            self.model.dof_damping[i] = 1e4
            self.model.qpos_spring[i] = self.data.qpos[i]

        for i in range(3, 6):
            self.model.dof_damping[i] = 1e5

    def adjust_robot_pose(self):
        """Adjust robot pose to avoid robot bodies stuck inside hfield or geoms. Make sure to call if env is updating the model.
        """
        # Make sure all kinematics are updated
        mj.mj_kinematics(self.model, self.data)
        # Check iteratively if robot is colliding with geom in XY and move robot up in Z
        lfoot_pos  = self.get_site_pose(self.feet_site_name[0])[0:3]
        rfoot_pos  = self.get_site_pose(self.feet_site_name[1])[0:3]
        z_deltas = []
        for (x,y,z) in [lfoot_pos, rfoot_pos]:
            box_id, heel_hgt = self.geom_generator.check_step(x - 0.1, y, 0)
            box_id, toe_hgt  = self.geom_generator.check_step(x + 0.1, y, 0)
            z_hgt = max(heel_hgt, toe_hgt)
            z_deltas.append((z_hgt-z))
        base_position = self.get_base_position()
        delta = max(z_deltas)
        base_position[2] += delta + 1e-3
        self.set_base_position(base_position)

    def release(self):
        """Zero stiffness/damping for base 6DOF
        """
        for i in range(3):
            self.model.jnt_stiffness[i] = 0
            self.model.dof_damping[i] = 0

        for i in range(3, 6):
            self.model.dof_damping[i] = 0

    def viewer_init(self, width=1280, height=960, camera_id=-1):
        self.viewer = MujocoViewer(self.model, self.data, self.reset_qpos, width=width, \
            height=height, camera_id=camera_id)

    def viewer_render(self):
        assert not self.viewer is None, \
               f"{FAIL}viewer has not been initalized yet, can not render.{ENDC}"
        if self.viewer.is_alive:
            return self.viewer.render()
        else:
            raise RuntimeError(f"{FAIL}Error: Viewer not alive, can not check paused status. Check "
                f"that viewer has not been destroyed.{ENDC}")

    def viewer_paused(self):
        assert not self.viewer is None, \
               f"{FAIL}viewer has not been initalized yet, can not check paused status.{ENDC}"
        if self.viewer.is_alive:
            return self.viewer.paused
        else:
            raise RuntimeError(f"{FAIL}Error: Viewer not alive, can not check paused status. Check "
                f"that viewer has not been destroyed.{ENDC}")

    def init_renderer(self, offscreen: bool, height: int, width: int):
        """Initialized renderer class for rendering.
        For offscreen render: Need to change os environ before
        importing all librairies (including mujoco). This does not work alongside with MujocoViewer.

        import os
        gl_option = 'egl' or 'glx' or 'osmesa'
        os.environ['MUJOCO_GL']=gl_option

        OSMesa has problems see, https://github.com/deepmind/mujoco/issues/700
        Args: height and width of image size. Once set, cannot change.
        """
        if offscreen:
            self.renderer = mj.Renderer(self.model, height=height, width=width)
            # print("Verify the Gl context object, ", self.renderer._gl_context)
        else:
            self.renderer = MujocoRender(self.model, height=height, width=width)

    def get_depth_image(self, camera_name: str):
        """ Get depth image given camera name. To use depth offscreen rendering,
        first call this init_offscreen_renderer() at reset(). And then call this function.
        """
        if camera_name is None:
            raise RuntimeError("Specify a camera name.")
        self.renderer.enable_depth_rendering()
        self.renderer.update_scene(self.data, camera=camera_name)
        depth = self.renderer.render().copy()
        # Perform in-place operations with depth values
        # Shift nearest values to the origin.
        depth -= depth.min()
        # Scale by 2 mean distances of near rays.
        depth /= (2*depth[depth <= 1].mean())
        # Scale to [0, 255]
        depth = 255*np.clip(depth, 0, 1)
        return depth.astype(np.uint8)

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

    def get_body_adr(self, name: str):
        return mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, name)

    def get_simulation_time(self):
        return self.data.time

    def get_body_pose(self, name: str):
        """Get object pose by name

        Args:
            name (str): object name

        Returns:
            ndarray: pose [3xlinear, 4xquaternion]
        """
        pose = np.zeros(7)
        pose[:3] = self.data.body(name).xpos
        pose[3:] = self.data.body(name).xquat
        return pose

    def get_site_pose(self, name: str):
        """Get site pose by name

        Args:
            name (str): site name

        Returns:
            ndarray: pose [3xlinear, 4xquaternion]
        """
        pose = np.zeros(7)
        pose[:3] = self.data.site(name).xpos
        mj.mju_mat2Quat(pose[3:7], self.data.site(name).xmat)
        return pose

    def get_geom_pose(self, name: str):
        """Get geom pose by name

        Args:
            name (str): geom name

        Returns:
            ndarray: pose [3xlinear, 4xquaternion]
        """
        pose = np.zeros(7)
        pose[:3] = self.data.geom(name).xpos
        mj.mju_mat2Quat(pose[3:7], self.data.geom(name).xmat)
        return pose

    def get_geom_size(self, name: str):
        """Get geom size by name

        Args:
            name (str): geom name

        Returns:
            ndarray: size [3x]
        """
        return self.model.geom(name).size

    def get_relative_pose(self, pose1: np.ndarray, pose2: np.ndarray):
        """Computes relative pose of object2 in the frame of object1.

        Args:
            pose1 (np.ndarray): pose of object1
            pose2 (np.ndarray): pose of object2

        Returns:
            pose2_in_pose1: relative pose
        """
        conjugate_pose1 = np.zeros(7)
        mj.mju_negPose(conjugate_pose1[0:3], conjugate_pose1[3:7],
                       pose1[0:3], pose1[3:7])
        pose2_in_pose1 = np.zeros(7)
        mj.mju_mulPose(pose2_in_pose1[0:3], pose2_in_pose1[3:7],
                       conjugate_pose1[0:3], conjugate_pose1[3:7],
                       pose2[0:3], pose2[3:7])
        return pose2_in_pose1

    def get_body_velocity(self, name: str, local_frame=False):
        """Get body velocity by name.
        # TODO: helei, make this generic to take geom and site

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

    def get_body_acceleration(self, name: str, local_frame=False):
        """Get body acceleration by name

        Args:
            name (str): body name
            local_frame (bool, optional): Defaults to False.

        Returns:
            ndarray: velocity [3xlinear, 3xangular]
        """
        body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, name)
        velocity = np.zeros(6)
        mj.mj_objectAcceleration(self.model, self.data, mj.mjtObj.mjOBJ_BODY, body_id, velocity, local_frame)
        tmp = velocity[3:6].copy()
        velocity[3:6] = velocity[0:3]
        velocity[0:3] = tmp
        return velocity

    def get_body_contact_force(self, name: str):
        """Get sum of contact forces at the named body in global frame

        Args:
            name (str): body name

        Returns:
            ndarray: sum of all wrenches acting on the body
        """
        body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, name)
        # Sum over all contact wrenches over possible geoms within the body
        total_wrench = np.zeros(6)
        contact_points = 0
        for contact_id, contact_struct in enumerate(self.data.contact):
            if body_id == self.model.geom_bodyid[contact_struct.geom1] or \
               body_id == self.model.geom_bodyid[contact_struct.geom2]:
                contact_points += 1
                contact_wrench_local = np.zeros(6)
                mj.mj_contactForce(self.model, self.data, contact_id, contact_wrench_local)
                contact_wrench_global = np.zeros(6)
                mj.mju_transformSpatial(contact_wrench_global, contact_wrench_local, True,
                                        self.data.xpos[body_id],
                                        self.data.contact[contact_id].pos,
                                        self.data.contact[contact_id].frame)
                if body_id == self.model.geom_bodyid[contact_struct.geom1]:
                    # This body is exerting forces onto geom2, substract from the sum.
                    total_wrench -= contact_wrench_global
                elif body_id == self.model.geom_bodyid[contact_struct.geom2]:
                    # This body is taking forces from geom1, add into the sum.
                    total_wrench += contact_wrench_global
        # Since condim=3, let's keep XYZ for now
        return total_wrench[:3]

    def get_body_mass(self, name: str = None):
        # If name is None, return all body masses
        if name:
            return self.model.body(name).mass
        else:
            return copy.deepcopy(self.model.body_mass)

    def get_body_ipos(self, name: str = None):
        # If name is None, return all body masses
        if name:
            return self.model.body(name).ipos
        else:
            return copy.deepcopy(self.model.body_ipos)

    def get_dof_damping(self, name: str = None):
        if name:
            return self.model.joint(name).damping
        else:
            return copy.deepcopy(self.model.dof_damping)

    def get_geom_friction(self, name: str = None):
        if name:
            return self.model.geom(name).friction
        else:
            return copy.deepcopy(self.model.geom_friction)

    def set_joint_position(self, position: np.ndarray):
        assert position.shape == (self.num_joints,), \
               f"{FAIL}set_joint_position got array of shape {position.shape} but " \
               f"should be shape ({self.num_joints},).{ENDC}"
        self.data.qpos[self.joint_position_inds] = position
        mj.mj_forward(self.model, self.data)

    def set_joint_velocity(self, velocity: np.ndarray):
        assert velocity.shape == (self.num_joints,), \
               f"{FAIL}set_joint_velocity got array of shape {velocity.shape} but " \
               f"should be shape ({self.num_joints},).{ENDC}"
        self.data.qvel[self.joint_velocity_inds] = velocity
        mj.mj_forward(self.model, self.data)

    def set_motor_position(self, position: np.ndarray):
        assert position.shape == (self.num_actuators,), \
               f"{FAIL}set_motor_position got array of shape {position.shape} but " \
               f"should be shape ({self.num_actuators},).{ENDC}"
        self.data.qpos[self.motor_position_inds] = position
        mj.mj_forward(self.model, self.data)

    def set_motor_velocity(self, velocity: np.ndarray):
        assert velocity.shape == (self.num_actuators,), \
               f"{FAIL}set_motor_velocity got array of shape {velocity.shape} but " \
               f"should be shape ({self.num_actuators},).{ENDC}"
        self.data.qvel[self.motor_velocity_inds] = velocity
        mj.mj_forward(self.model, self.data)

    def set_base_position(self, position: np.ndarray):
        assert position.shape == (3,), \
               f"{FAIL}set_base_position got array of shape {position.shape} but " \
               f"should be shape (3,).{ENDC}"
        self.data.qpos[self.base_position_inds] = position
        mj.mj_forward(self.model, self.data)

    def set_base_linear_velocity(self, velocity: np.ndarray):
        assert velocity.shape == (3,), \
               f"{FAIL}set_base_linear_velocity got array of shape {velocity.shape} but " \
               f"should be shape (3,).{ENDC}"
        self.data.qvel[self.base_linear_velocity_inds] = velocity
        mj.mj_forward(self.model, self.data)

    def set_base_orientation(self, quat: np.ndarray):
        assert quat.shape == (4,), \
               f"{FAIL}set_base_orientation got array of shape {quat.shape} but " \
               f"should be shape (4,).{ENDC}"
        self.data.qpos[self.base_orientation_inds] = quat
        mj.mj_forward(self.model, self.data)

    def set_base_angular_velocity(self, velocity: np.ndarray):
        assert velocity.shape == (3,), \
               f"{FAIL}set_base_angular_velocity got array of shape {velocity.shape} but " \
               f"should be shape (3,).{ENDC}"
        self.data.qvel[self.base_angular_velocity_inds] = velocity
        mj.mj_forward(self.model, self.data)

    def set_body_mass(self, mass: float | int | np.ndarray, name: str = None):
        # If name is None, expect setting all masses
        if name:
            assert isinstance(mass, (float, int)), \
                f"{FAIL}set_body_mass got a {type(mass)} instead of a single float when setting mass " \
                f"for single body {name}.{ENDC}"
            self.model.body(name).mass = mass
        else:
            assert mass.shape == (self.model.nbody,), \
                f"{FAIL}set_body_mass got array of shape {mass.shape} but should be shape " \
                f"({self.model.nbody},).{ENDC}"
            self.model.body_mass = mass

    def set_body_ipos(self, ipos: np.ndarray, name: str = None):
        if name:
            assert ipos.shape == (3,), \
                f"{FAIL}set_body_ipos got array of shape {ipos.shape} when setting ipos for " \
                f"single body {name} but should be shape (3,).{ENDC}"
            self.model.body(name).ipos = ipos
        else:
            assert ipos.shape == (self.model.nbody, 3), \
                f"{FAIL}set_body_mass got array of shape {ipos.shape} but should be shape " \
                f"({self.model.nbody}, 3).{ENDC}"
            self.model.body_ipos = ipos

    def set_dof_damping(self, damp: float | int | np.ndarray, name: str = None):
        if name:
            num_dof = len(self.model.joint(name).damping)
            if num_dof == 1:
                assert isinstance(damp, (float, int)), \
                    f"{FAIL}set_dof_damping got a {type(damp)} when setting damping for single dof " \
                    f"{name} but should be a float or int.{ENDC}"
            else:
                assert damp.shape == (num_dof,), \
                    f"{FAIL}set_dof_damping got array of shape {damp.shape} when setting damping " \
                    f"for single dof {name} but should be shape ({num_dof},).{ENDC}"
            self.model.joint(name).damping = damp
        else:
            assert damp.shape == (self.model.nv,), \
                f"{FAIL}set_dof_damping got array of shape {damp.shape} when setting all joint " \
                f"dofs but should be shape ({self.model.nv},).{ENDC}"
            self.model.dof_damping = damp

    def set_geom_friction(self, fric: np.ndarray, name: str = None):
        if name:
            assert fric.shape == (3, ), \
                f"{FAIL}set_geom_friction got array of shape {fric.shape} when setting friction " \
                f"for single geom {name} but should be shape (3,).{ENDC}"
            self.model.geom(name).friction = fric
        else:
            assert fric.shape == (self.model.ngeom, 3), \
                f"{FAIL}set_geom_friction got array of shape {fric.shape} when setting all geom " \
                f"friction but should be shape ({self.model.ngeom}, 3).{ENDC}"
            self.model.geom_friction = fric
    """The following setters are meant during simulation or reset, so should not advance simulation by itself.
    """
    def set_geom_pose(self, name: str, pose: np.ndarray):
        self.model.geom(name).pos = pose[0:3]
        self.model.geom(name).quat = pose[3:7]

    def set_geom_size(self, name: str, size: np.ndarray):
        self.model.geom(name).size = size

    def set_geom_color(self, name: str, color: np.ndarray):
        self.model.geom(name).rgba = color

    def set_body_pose(self, name: str, pose: np.ndarray):
        self.data.body(name).xpos = pose[0:3]
        self.data.body(name).xquat = pose[3:7]
