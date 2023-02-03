import numpy as np
from env.util.quaternion import *

def kernel(x):
  return np.exp(-x)

def compute_reward(self, action):
    assert hasattr(self, "clock"), \
        f"Environment {self.__class__.__name__} does not have a clock object"
    assert self.clock is not None, \
        f"Clock has not been initialized, is still None"
    assert self.clock_type == "von_mises", \
        f"locomotion_vonmises_clock_reward should be used with von mises clock type, but clock type" \
        f"is {self.clock_type}."

    q = {}

    ### Cyclic foot force/velocity reward ###
    # Get foot cost clock weightings, linear replacement for the von mises function. There are two separate clocks
    # with different transition timings for the force/velocity cost and the position (foot height) cost.
    # Force/velocity cost transitions between stance and swing quicker to try and make sure get full swing phase in there,
    # especially important for higher speeds where we try to enforce longer swing time for aerial phase.
    # However, don't need foot height to reach clearance level as quickly, so can use smoother transition (higher percent_trans)
    l_force, r_force = self.clock.von_mises()
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

    q["left_force"] = l_force * np.abs(feet_force[0])
    q["right_force"] = r_force * np.abs(feet_force[1])
    q["left_speed"] = l_stance * feet_vel[0]
    q["right_speed"] = r_stance * feet_vel[1]

    ### Speed rewards ###
    base_vel = self.sim.get_body_velocity(self.sim.base_body_name)
    x_vel = np.abs(base_vel[0] - self.x_velocity)
    y_vel = np.abs(base_vel[1] - self.y_velocity)
    # We have deadzones around the speed reward since it is impossible (and we actually don't want)
    # for base velocity to be constant the whole time.
    if x_vel < 0.05:
        x_vel = 0
    if y_vel < 0.05:
        y_vel = 0
    q["x_vel"] = x_vel
    q["y_vel"] = y_vel

    ### Orientation rewards (base and feet) ###
    base_pose = self.sim.get_body_pose(self.sim.base_body_name)
    target_quat = np.array([1, 0, 0, 0])
    command_quat = euler2quat(z = self.orient_add, y = 0, x = 0)
    target_quat = quaternion_product(target_quat, command_quat)
    orientation_error = quaternion_similarity(base_pose[3:], target_quat)
    # Deadzone around quaternion as well
    if orientation_error < 5e-3:
        orientation_error = 0

    # Foor orientation target in global frame. Heuristic hard coded value to be flat all the time.
    # If we change the turn command (self.orient_add) then need to rotate the foot orient target
    # as well. NOTE: Should figure out Mujoco body rotations so don't have to do this.
    # NOTE: For Cassie this target is the same for both feet, for Digit I think some things might be
    # flipped between left and right.
    foot_orient_target = np.array([-0.24790886454547323, -0.24679713195445646, -0.6609396704367185, 0.663921021343526])
    if self.orient_add != 0:
        iquaternion = inverse_quaternion(command_quat)
        foot_orient_target = quaternion_product(iquaternion, foot_orient_target)
    foot_orientation_error = quaternion_similarity(foot_orient_target, feet_pose[0, 3:]) + \
                             quaternion_similarity(foot_orient_target, feet_pose[1, 3:])
    q["orientation"] = orientation_error + foot_orientation_error

    ### Hop symmetry reward (keep feet equidistant) ###
    period_shifts = self.clock.get_period_shifts()
    # lpos, rpos = self.get_info('robot_foot_positions', local=True)
    rel_foot_pos = np.subtract(feet_pose[:, 0:3], base_pose[0:3])
    # lpos = np.array([rel_foot_pos[0], rel_foot_pos[2]])
    # rpos = np.array([rel_foot_pos[0], rel_foot_pos[2]])
    xdif = np.sqrt(np.power(rel_foot_pos[0, [0, 2]] - rel_foot_pos[1, [0, 2]], 2).sum())
    pdif = np.exp(-5 * np.abs(np.sin(np.pi * (period_shifts[0] - period_shifts[1]))))
    q['hop_symmetry'] = pdif * xdif

    ### Sim2real stability rewards ###
    base_acc = self.sim.get_body_acceleration(self.sim.base_body_name)
    q["stable_base"] = np.abs(base_vel[3:]).sum() + np.abs(base_acc[0:2]).sum()
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

# Termination condition: If orientation too far off terminate
def compute_done(self):
    base_quat = self.sim.get_body_pose(self.sim.base_body_name)[3:]
    target_quat = np.array([1, 0, 0, 0])
    command_quat = euler2quat(z = self.orient_add, y = 0, x = 0)
    target_quat = quaternion_product(target_quat, command_quat)
    orientation_error = 3 * quaternion_similarity(base_quat, target_quat)
    if np.exp(-orientation_error) < 0.8:
        return True
    else:
        return False
