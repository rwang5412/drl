import numpy as np
from env.util.quaternion import *

def kernel(x):
  return np.exp(-x)

def compute_reward(self, action):
    assert hasattr(self, "clock"), \
        f"Environment {self.__class__.__name__} does not have a clock object"
    assert self.clock is not None, \
        f"Clock has not been initialized, is still None"
    assert self.clock_type == "linear", \
        f"locomotion_linear_clock_reward should be used with linear clock type, but clock type is" \
        f"{self.clock_type}."

    # Weighting dictionary to make it easier to see all reward components and change their
    # individual weightings
    q = {}
    w = {}
    w['x_vel']                  = 0.1
    w['y_vel']                  = 0.1
    w['base_orientation']       = 0.1
    w['l_foot_cost_forcevel']   = 0.1
    w['r_foot_cost_forcevel']   = 0.1
    w['l_foot_cost_pos']        = 0.1
    w['r_foot_cost_pos']        = 0.1
    w['l_foot_orientation']     = 0.05
    w['r_foot_orientation']     = 0.05
    w['hiproll_cost']           = 0.05
    w['hipyaw_vel']             = 0.05
    w['base_transacc']          = 0.025
    w['base_rotvel']            = 0.025
    w['ctrl_penalty']           = 0.025
    w['trq_penalty']            = 0.025

    # Just in case we made a mistake in the weighting above, make sure that weightings
    # are normalized so sum will equal to 1
    total = sum(w.values())
    for name in w:
        w[name] = w[name] / total

    ### Cyclic foot force/velocity reward ###
    # Get foot cost clock weightings, linear replacement for the von mises function. There are two separate clocks
    # with different transition timings for the force/velocity cost and the position (foot height) cost.
    # Force/velocity cost transitions between stance and swing quicker to try and make sure get full swing phase in there,
    # especially important for higher speeds where we try to enforce longer swing time for aerial phase.
    # However, don't need foot height to reach clearance level as quickly, so can use smoother transition (higher percent_trans)
    l_force, r_force = self.clock.linear_clock(percent_transition = 0.2)
    l_swing, r_swing = self.clock.linear_clock(percent_transition = 0.7)
    l_stance = 1 - l_force
    r_stance = 1 - r_force
    # Ok to assume that left foot name comes first?
    # Probably fine to assume that there are only 2 feet
    feet_force = np.zeros(2)
    feet_vel = np.zeros(2)
    feet_pose = np.zeros((2, 7))
    for i in range(2):
        feet_force[i] = np.linalg.norm(self.sim.get_body_contact_force(self.sim.feet_body_name[i]))
        feet_vel[i] = np.linalg.norm(self.sim.get_body_velocity(self.sim.feet_body_name[i])[0:3])
        feet_pose[i, :] = self.sim.get_body_pose(self.sim.feet_body_name[i])

    if self.speed <= 1:
        des_foot_height = 0.1
    elif 1 < self.speed <= 3:
        des_foot_height = 0.1 + 0.2 * (self.speed - 1) / 2
    else:
        des_foot_height = 0.3

    l_force_cost = np.abs(feet_force[0]) / 75
    r_force_cost = np.abs(feet_force[1]) / 75
    l_height_cost = 40 * (des_foot_height - feet_pose[0, 2])**2
    r_height_cost = 40 * (des_foot_height - feet_pose[1, 2])**2

    q["l_foot_cost_forcevel"] = l_force * l_force_cost + l_stance * feet_vel[0]
    q["r_foot_cost_forcevel"] = r_force * r_force_cost + r_stance * feet_vel[1]
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
    command_quat = euler2quat(z = self.orient_add, y = 0, x = 0)
    target_quat = quaternion_product(target_quat, command_quat)
    orientation_error = quaternion_similarity(base_quat, target_quat)
    # Deadzone around quaternion as well
    if orientation_error < 5e-3:
        orientation_error = 0
    else:
        orientation_error *= 30
    q["x_vel"] = 2 * x_vel
    q["y_vel"] = 2 * y_vel
    q["base_orientation"] = orientation_error

    ### Foot orientation rewards ###
    # Foor orientation target in global frame. Heuristic hard coded value to be flat all the time.
    # If we change the turn command (self.orient_add) then need to rotate the foot orient target
    # as well. NOTE: Should figure out Mujoco body rotations so don't have to do this.
    # NOTE: For Cassie this target is the same for both feet, for Digit I think some things might be
    # flipped between left and right.
    foot_orient_target = np.array([-0.24790886454547323, -0.24679713195445646, -0.6609396704367185, 0.663921021343526])
    if self.orient_add != 0:
        iquaternion = inverse_quaternion(command_quat)
        foot_orient_target = quaternion_product(iquaternion, foot_orient_target)
    q["l_foot_orientation"] = 20 * quaternion_similarity(foot_orient_target, feet_pose[0, 3:])
    q["r_foot_orientation"] = 20 * quaternion_similarity(foot_orient_target, feet_pose[1, 3:])

    ### Stable base reward terms.  Don't want base to rotate or accelerate too much ###
    base_acc = self.sim.get_body_acceleration(self.sim.base_body_name)
    q["base_rotvel"] = 2 * np.linalg.norm(base_vel[3:])
    q["base_transacc"] = np.linalg.norm(base_acc[0:2]) # Don't care about z acceleration

    ### Sim2real stability rewards ###
    motor_vel = self.sim.get_motor_velocity()
    q["hiproll_cost"] = (np.abs(motor_vel[0]) + np.abs(motor_vel[5])) / 3
    q["hipyaw_vel"] = (np.abs(motor_vel[1]) + np.abs(motor_vel[6]))
    if self.last_action is not None:
        q["ctrl_penalty"] = 5 * sum(np.abs(self.last_action - action)) / len(action)
    else:
        q["ctrl_penalty"] = 0
    torque = self.sim.get_torque()
    q["trq_penalty"] = 0.05 * sum(np.abs(torque)) / len(torque)

    ### Add up all reward components ###
    self.reward = 0
    for name in w:
        self.reward += w[name] * kernel(q[name])

    return self.reward

# Termination condition: If reward is too low or height is too low (cassie fell down) terminate
def compute_done(self):
    base_height = self.sim.get_body_pose(self.sim.base_body_name)[2]
    if base_height < 0.4 or self.reward < 0.4:
        return True
    else:
        return False
