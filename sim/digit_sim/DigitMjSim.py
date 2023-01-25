import mujoco as mj
import numpy as np
import pathlib

from sim import GenericSim, MujocoViewer

WARNING = '\033[93m'
ENDC = '\033[0m'

class DigitMjSim(GenericSim):
  """Wrapper for Digit Mujoco.
  """
  def __init__(self) -> None:
    super().__init__()
    model_path = pathlib.Path(__file__).parent.resolve() / "digit-v3-new.xml"
    self.model = mj.MjModel.from_xml_path(str(model_path))
    self.data = mj.MjData(self.model)
    self.viewer = None

    self.motor_position_inds = [7, 8, 9, 14, 18, 23, 30, 31, 32, 33, 34, 35, 36, 41, 45, 50, 57, 58, 59, 60]
    self.motor_velocity_inds = [6, 7, 8, 12, 16, 20, 26, 27, 28, 29, 30, 31, 32, 36, 40, 44, 50, 51, 52, 53]
    self.joint_position_inds = [15, 16, 17, 28, 29, 42, 43, 44, 55, 56]
    self.joint_velocity_inds = [13, 14, 15, 24, 25, 37, 38, 39, 48, 49]

    self.base_position_inds = [0, 1, 2]
    self.base_orientation_inds = [3, 4, 5, 6]
    self.base_linear_velocity_inds = [0, 1, 2]
    self.base_angular_velocity_inds = [3, 4, 5]

    self.num_actuators = self.model.nu
    self.num_joints = len(self.joint_position_inds)
    
    # TODO: helei, We might push this into env
    self.kp = np.array([200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0,
                        200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0])
    self.kd = np.array([10.0, 10.0, 20.0, 20.0, 7.0, 7.0, 10.0, 10.0, 10.0, 10.0,
                        10.0, 10.0, 20.0, 20.0, 7.0, 7.0, 10.0, 10.0, 10.0, 10.0])
    
    self.reset_qpos = np.array([0, 0, 1,  1, 0 ,0 ,0,
      3.33020155e-01, -2.66178730e-02, 1.92369587e-01, 
      9.93409734e-01, -1.04126145e-03, 1.82534311e-03, 1.14597921e-01,
      2.28971047e-01, 1.48527831e-03, -2.31455693e-01, 4.55857916e-04, 
      -1.29734322e-02,  
      9.89327705e-01, 1.45524756e-01, 1.73630859e-03, 7.08678995e-03, 
      -2.03852305e-02,
      9.88035432e-01, 1.53876629e-01, 6.59769560e-05, 1.03905844e-02,
      -3.52778547e-03, -5.54992074e-02, 
      -1.05542715e-01, 8.94852532e-01, -8.63756398e-03, 3.44780280e-01,
      -3.33020070e-01, 2.66195360e-02, -1.92382190e-01,
      9.93409659e-01, 1.04481446e-03, 1.82489637e-03, -1.14598546e-01,
      -2.28971188e-01, -1.48636971e-03, 2.31454977e-01, -4.53425792e-04,
      1.41299940e-02,
      9.89323654e-01, -1.45521550e-01, 1.80177609e-03, -7.67726135e-03,
      1.93767895e-02,
      9.88041418e-01, -1.53872399e-01, 1.26073837e-05, -9.87117790e-03,
      2.36874210e-03, 5.55559678e-02,
      1.05444698e-01, -8.94890429e-01, 8.85979401e-03, -3.44723293e-01
    ])
    
    self.offset = self.reset_qpos[self.motor_position_inds]

  def reset(self, qpos: np.ndarray=None):
    if qpos:
      assert len(qpos) == self.model.nq, f"reset qpos len={len(qpos)}, but should be {self.model.nq}"
      self.data.qpos = qpos
    else:
      self.data.qpos = self.reset_qpos
    mj.mj_forward(self.model, self.data)
  
  def sim_forward(self, dt: float=None):
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

  def get_base_position(self):
      return self.data.qpos[self.base_position_inds]

  def get_base_linear_velocity(self):
      return self.data.qvel[self.base_linear_velocity_inds]

  def get_base_orientation(self):
      return self.data.qvel[self.base_orientation_inds]

  def get_base_angular_velocity(self):
      return self.data.qvel[self.base_angular_velocity_inds]

  def get_torque(self):
    return self.data.ctrl[:]

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

  def set_base_position(self, position: np.ndarray):
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
