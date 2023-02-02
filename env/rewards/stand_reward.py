import numpy as np
from env.util.quaternion import *

def kernel(x):
  return np.exp(-x)

def compute_reward(self, action):
    # Weighting dictionary to make it easier to see all reward components and change their
    # individual weightings
    q = {}
    w = {}
    w['height_penalty']         = 0.3
    w['pelvis_orientation']     = 0.1
    w['l_foot_orientation']     = 0.05
    w['r_foot_orientation']     = 0.05
    w['foot_vel_penalty']       = 0.2
    w['motor_vel_penalty']      = 0.1
    w['ctrl_penalty']           = 0.1
    w['trq_penalty']            = 0.1

    # Just in case we made a mistake in the weighting above, make sure that weightings
    # are normalized so sum will equal to 1
    total = sum(w.values())
    for name in w:
        w[name] = w[name] / total

    ### Height penalty, match the desired standing height ###
    pelvis_pose = self.sim.get_body_pose(self.sim.base_body_name)
    q['height_penalty'] = 3 * np.abs(pelvis_pose[2] - 0.9)

    ### Orientation rewards, pelvis and feet ###
    # Ok to assume that left foot name comes first?
    # Probably fine to assume that there are only 2 feet
    feet_vel = np.zeros(2)
    feet_pose = np.zeros((2, 7))
    for i in range(2):
        feet_vel[i] = np.linalg.norm(self.sim.get_body_velocity(self.sim.feet_body_name[i])[0:3])
        feet_pose[i, :] = self.sim.get_body_pose(self.sim.feet_body_name[i])
    foot_orient_target = np.array([-0.24790886454547323, -0.24679713195445646, -0.6609396704367185, 0.663921021343526])
    q["l_foot_orientation"] = 20 * (1 - np.inner(foot_orient_target, feet_pose[0, 3:]) ** 2)
    q["r_foot_orientation"] = 20 * (1 - np.inner(foot_orient_target, feet_pose[1, 3:]) ** 2)
    pelvis_orient_target = np.array([1, 0, 0, 0])
    q["pelvis_orientation"] = 1 - np.inner(pelvis_pose[3:], pelvis_orient_target) ** 2

    ### Static rewards. Want feet and motor velocities to be zero ###
    motor_vel = self.sim.get_motor_velocity()
    q['motor_vel_penalty'] = np.linalg.norm(motor_vel)
    q['foot_vel_penalty'] = 2 * np.sum(feet_vel)

    ### Control rewards ###
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

# Termination condition: If height is too low (cassie fell down) terminate
def compute_done(self):
    pelvis_height = self.sim.get_body_pose(self.sim.base_body_name)[2]
    if pelvis_height < 0.4:
        return True
    else:
        return False
