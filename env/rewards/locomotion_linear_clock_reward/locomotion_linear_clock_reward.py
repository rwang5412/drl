import numpy as np

from env.util.quaternion import *
from util.colors import *

def kernel(x):
  return np.exp(-x)

def compute_reward(self, action):
    assert hasattr(self, "clock"), \
        f"{FAIL}Environment {self.__class__.__name__} does not have a clock object.{ENDC}"
    assert self.clock is not None, \
        f"{FAIL}Clock has not been initialized, is still None.{ENDC}"
    assert self.clock_type == "linear", \
        f"{FAIL}locomotion_linear_clock_reward should be used with linear clock type, but clock " \
        f"type is {self.clock_type}.{ENDC}"

    q = {}

    ### Cyclic foot force/velocity reward ###
    # Get foot cost clock weightings, linear replacement for the von mises function. There are two
    # separate clocks with different transition timings for the force/velocity cost and the position
    # (foot height) cost. Force/velocity cost transitions between stance and swing quicker to try
    # and make sure get full swing phase in there, especially important for higher speeds where we
    # try to enforce longer swing time for aerial phase. However, don't need foot height to reach
    # clearance level as quickly, so can use smoother transition (higher percent_trans)
    l_force, r_force = self.clock.linear_clock(percent_transition = 0.2)
    l_swing, r_swing = self.clock.linear_clock(percent_transition = 0.7)
    l_stance = 1 - l_force
    r_stance = 1 - r_force

    feet_force = {}
    feet_vel = {}
    feet_pose = {}
    for foot_name in self.sim.feet_body_name:
        vel = np.linalg.norm(self.sim.get_body_velocity(foot_name)[0:3])
        force = np.linalg.norm(self.sim.get_body_contact_force(foot_name))
        side = "left" if "left" in foot_name else "right"
        feet_force[f"{side} foot"] = force
        feet_vel[f"{side} foot"] = vel
    for foot_name in self.sim.feet_site_name:
        pose = self.sim.get_site_pose(foot_name)
        side = "left" if "left" in foot_name else "right"
        feet_pose[f"{side} foot"] = pose

    if self.x_velocity <= 1:
        des_foot_height = 0.1
    elif 1 < self.x_velocity <= 3:
        des_foot_height = 0.1 + 0.2 * (self.x_velocity - 1) / 2
    else:
        des_foot_height = 0.3

    l_force_cost = np.abs(feet_force["left foot"]) / 75
    r_force_cost = np.abs(feet_force["right foot"]) / 75
    l_height_cost = (des_foot_height - feet_pose["left foot"][2])**2
    r_height_cost = (des_foot_height - feet_pose["right foot"][2])**2

    q["l_foot_cost_forcevel"] = l_force * l_force_cost + l_stance * feet_vel["left foot"]
    q["r_foot_cost_forcevel"] = r_force * r_force_cost + r_stance * feet_vel["right foot"]
    q["l_foot_cost_pos"] = l_swing * l_height_cost
    q["r_foot_cost_pos"] = r_swing * r_height_cost

    ### CoM rewards (desired speed, orientation) ###
    base_vel = self.sim.get_body_velocity(self.sim.base_body_name)
    x_vel = np.abs(base_vel[0] - self.x_velocity)
    y_vel = np.abs(base_vel[1] - self.y_velocity)
    # We have deadzones around the speed reward since it is impossible (and we actually don't want)
    # for base velocity to be constant the whole time.
    if x_vel < 0.05:
        x_vel = 0
    if y_vel < 0.05:
        y_vel = 0

    base_quat = self.sim.get_body_pose(self.sim.base_body_name)[3:]
    target_quat = np.array([1, 0, 0, 0])
    if self.orient_add != 0:
        command_quat = euler2quat(z = self.orient_add, y = 0, x = 0)
        target_quat = quaternion_product(target_quat, command_quat)
    orientation_error = quaternion_distance(base_quat, target_quat)
    # Deadzone around quaternion as well
    if orientation_error < 5e-3:
        orientation_error = 0
    q["x_vel"] = x_vel
    q["y_vel"] = y_vel
    q["base_orientation"] = orientation_error

    ### Foot orientation rewards ###
    # Foor orientation target in global frame. Want to be flat and face same direction as base all
    # the time. So compare to the same orientation target as the base.
    q["l_foot_orientation"] = quaternion_distance(target_quat, feet_pose["left foot"][3:])
    q["r_foot_orientation"] = quaternion_distance(target_quat, feet_pose["right foot"][3:])

    ### Stable base reward terms.  Don't want base to rotate or accelerate too much ###
    base_acc = self.sim.get_body_acceleration(self.sim.base_body_name)
    q["base_rotvel"] = np.linalg.norm(base_vel[3:])
    q["base_transacc"] = np.linalg.norm(base_acc[0:2]) # Don't care about z acceleration

    ### Sim2real stability rewards ###
    motor_vel = self.sim.get_motor_velocity()
    q["hiproll_cost"] = np.abs(motor_vel[0]) + np.abs(motor_vel[5])
    q["hipyaw_vel"] = np.abs(motor_vel[1]) + np.abs(motor_vel[6])
    if self.last_action is not None:
        q["ctrl_penalty"] = sum(np.abs(self.last_action - action)) / len(action)
    else:
        q["ctrl_penalty"] = 0
    torque = self.sim.get_torque()
    q["trq_penalty"] = sum(np.abs(torque)) / len(torque)

    ### Add up all reward components ###
    self.reward = 0
    for name in q:
        self.reward += self.reward_weight[name]["weighting"] * \
                       kernel(self.reward_weight[name]["scaling"] * q[name])

    return self.reward

# Termination condition: If reward is too low or height is too low (cassie fell down) terminate
def compute_done(self):
    base_height = self.sim.get_body_pose(self.sim.base_body_name)[2]
    if base_height < 0.4 or self.reward < 0.4:
        return True
    else:
        return False
