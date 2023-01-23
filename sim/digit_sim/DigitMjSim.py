import mujoco as mj
import numpy as np
import pathlib

from ..GenericSim import GenericSim
from ..MujocoViewer import MujocoViewer

class DigitMjSim(GenericSim):
  """Wrapper for Digit Mujoco.
  """
  def __init__(self) -> None:
    super().__init__()
    model_path = pathlib.Path(__file__).parent.resolve() / "digit-v3-new.xml"
    self.model = mj.MjModel.from_xml_path(str(model_path))
    self.data = mj.MjData(self.model)
    self.viewer = None

    self.motor_position_inds=[7, 8, 9, 14, 18, 23, 30, 31, 32, 33, 34, 35, 36, 41, 45, 50, 57, 58, 59, 60]
    self.motor_velocity_inds=[6, 7, 8, 12, 16, 20, 26, 27, 28, 29, 30, 31, 32, 36, 40, 44, 50, 51, 52, 53]
    self.joint_position_inds=[15, 16, 17, 28, 29, 42, 43, 44, 55, 56]
    self.joint_velocity_inds=[13, 14, 15, 24, 25, 37, 38, 39, 48, 49]

    self.base_position_inds = [0, 1, 2]
    self.base_orientation_inds = [3, 4, 5, 6]
    self.base_linear_velocity_inds = [0, 1, 2]
    self.base_angular_velocity_inds = [3, 4, 5]

    self.num_actuators = self.model.nu
    self.num_joints = len(self.joint_position_inds)
    
    self.offset = np.array([0.0045, 0.0, 0.4973, -1.1997, -1.5968, 0.0045, 0.0, 0.4973, -1.1997, -1.5968])
    
    # TODO: helei, We might push this into env
    self.kp = np.array([100,  100,  88,  96,  50, 100, 100,  88,  96,  50])
    self.kd = np.array([10.0, 10.0, 8.0, 9.6, 5.0, 10.0, 10.0, 8.0, 9.6, 5.0])
    
    # TODO: helei, need to actually write the conrod q correctly. Need IK for this.
    self.reset_qpos = np.array([0, 0, 1, 1, 0, 0, 0,
                      0.332, -0.00524161, 0.178407, 1,0,0,0, 0.21412, 0.00520115, -0.228917,
                      0.0544359, -0.000953898, 1,0,0,0, 0.00220685, 1,0,0,0, -0.0521339, 0.0516071,
                      -0.332, 0.00523932, -0.178411, 1,0,0,0, -0.214114, -0.00520115, 0.222871,
                      -0.0544391, 0.000977788, 1,0,0,0, -0.002214, 1,0,0,0, 0.0521441, -0.0516029,
                      -0.106437, 0.89488, -0.00867663, 0.344684,
                      0.106339, -0.894918, 0.00889888, -0.344627
                      ])

  def reset(self, qpos: np.ndarray=None):
    if qpos:
      assert len(qpos) == self.model.nq, f"reset qpos with {len(qpos)}, but should be {self.model.nq}"
      self.data.qpos = qpos
    else:
      self.data.qpos = self.reset_qpos
  
  def sim_forward(self, dt: float=None):
    if dt:
      num_steps = int(dt / self.model.opt.timestep)
      WARNING = '\033[93m'
      ENDC = '\033[0m'
      if num_steps * self.model.opt.timestep != dt:
        raise RuntimeError(f"{WARNING}Warning: {dt} does not fit evenly within the sim timestep of"
            f" {self.model.opt.timestep}, simulating forward"
            f" {num_steps * self.model.opt.timestep}s instead.{ENDC}") 
    else:
      num_steps = 1
    mj.mj_step(self.model, self.data, nstep=num_steps)

  def set_torque(self, torque: np.ndarray):
    """Set torque to simulator.

    Args:
        torque (np.ndarray, optional): Torque values for actuated joints. Defaults to None.
    """
    assert torque.ndim == 1, \
            f"set_torque did not receive a 1 dimensional array"
    assert len(torque) == self.model.nu, \
            f"set_torque did not receive array of size {self.model.nu}"
    self.data.ctrl[:] = torque

  def set_PD(self, 
             p: np.ndarray, 
             d: np.ndarray, 
             kp: np.ndarray, 
             kd: np.ndarray):
    assert p.ndim == 1, \
            f"set_PD P_targ was not a 1 dimensional array"
    assert d.ndim == 1, \
            f"set_PD D_targ was not a 1 dimensional array"
    assert kp.ndim == 1, \
            f"set_PD P_gain was not a 1 dimensional array"
    assert kd.ndim == 1, \
            f"set_PD D_gain was not a 1 dimensional array"
    assert len(p) == self.model.nu, \
            f"set_PD P_targ was not array of size {self.model.nu}"
    assert len(d) == self.model.nu, \
            f"set_PD D_targ was not array of size {self.model.nu}"
    assert len(kp) == self.model.nu, \
            f"set_PD P_gain was not array of size {self.model.nu}"
    assert len(kd) == self.model.nu, \
            f"set_PD D_gain was not array of size {self.model.nu}"
    torque = kp * (p - self.data.qpos[self.motor_pos_inds]) + \
              kd * (d - self.data.qvel[self.motor_vel_inds])
    self.data.ctrl[:] = torque
    
  def hold(self):
    """Set stiffness/damping for base 6DOF so base is fixed
    """
    for i in range(3):
      self.model.jnt_stiffness[i] = 1e5
      self.model.dof_damping[i] = 1e4
      self.model.qpos_spring[i] = self.data.qpos[i]

    for i in range(3, 6):
      self.model.dof_damping[i] = 1e4

  def release(self):
    """Zero stiffness/damping for base 6DOF
    """
    for i in range(3):
        self.model.jnt_stiffness[i] = 0
        self.model.dof_damping[i] = 0

    for i in range(3, 6):
        self.model.dof_damping[i] = 0

  def viewer_init(self):
      self.viewer = MujocoViewer(self.model, self.data, self.reset_qpos)

  def viewer_render(self):
      if self.viewer.is_alive:
          self.viewer.render()
      else:
          raise RuntimeError("Error: Viewer not alive, can not render.")

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

  def get_base_translation(self):
      return self.data.qpos[self.base_position_inds]

  def get_base_linear_velocity(self):
      return self.data.qvel[self.base_linear_velocity_inds]

  def get_base_orientation(self):
      return self.data.qvel[self.base_orientation_inds]

  def get_base_angular_velocity(self):
      return self.data.qvel[self.base_angular_velocity_inds]

  def set_joint_position(self, position: np.ndarray):
      assert len(position) == self.num_joints, \
        f"set_joint_position got {len(position)} but should be {self.num_joints}."
      self.data.qpos[self.joint_position_inds] = position

  def set_joint_velocity(self, velocity: np.ndarray):
      assert len(velocity) == self.num_joints, \
        f"set_joint_position got {len(velocity)} but should be {self.num_joints}."
      self.data.qvel[self.joint_velocity_inds] = velocity

  def set_motor_position(self, position: np.ndarray):
      assert len(position) == self.num_actuators, \
        f"set_motor_position got {len(position)} but should be {self.num_actuators}."
      self.data.qpos[self.joint_velocity_inds] = position

  def set_motor_velocity(self, velocity: np.ndarray):
      assert len(velocity) == self.num_actuators, \
        f"set_motor_position got {len(velocity)} but should be {self.num_actuators}."
      self.data.qvel[self.motor_velocity_inds] = velocity

  def set_base_translation(self, position: np.ndarray):
      assert len(position) == 3, \
        f"set_base_translation got {len(position)} but should be 3."
      self.data.qpos[self.base_position_inds] = position

  def set_base_linear_velocity(self, velocity: np.ndarray):
      assert len(velocity) == 3, \
        f"set_base_linear_velocity got {len(velocity)} but should be 3."
      self.data.qvel[self.base_linear_velocity_inds] = velocity

  def set_base_orientation(self, quat: np.ndarray):
      assert len(quat) == 4, \
        f"set_base_orientation got {len(quat)} but should be 4."
      self.data.qpos[self.base_orientation_inds] = quat

  def set_base_angular_velocity(self, velocity: np.ndarray):
      assert len(velocity) == 3, \
        f"set_base_angular_velocity got {len(velocity)} but should be 3."
      self.data.qvel[self.base_angular_velocity_inds] = velocity
