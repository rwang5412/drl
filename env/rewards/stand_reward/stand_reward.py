import numpy as np
from env.util.quaternion import *

def kernel(x):
  return np.exp(-x)

def compute_reward(self, action):
    q = {}

    ### Height penalty, match the desired standing height ###
    base_pose = self.sim.get_body_pose(self.sim.base_body_name)
    q['height_penalty'] = np.abs(base_pose[2] - 0.9)

    ### Orientation rewards, base and feet ###
    # Ok to assume that left foot name comes first?
    # Probably fine to assume that there are only 2 feet
    feet_vel = np.zeros(2)
    feet_pose = np.zeros((2, 7))
    for i in range(2):
        feet_vel[i] = np.linalg.norm(self.sim.get_body_velocity(self.sim.feet_body_name[i])[0:3])
        feet_pose[i, :] = self.sim.get_body_pose(self.sim.feet_body_name[i])
    foot_orient_target = np.array([-0.24790886454547323, -0.24679713195445646, -0.6609396704367185, 0.663921021343526])
    q["l_foot_orientation"] = (1 - np.inner(foot_orient_target, feet_pose[0, 3:]) ** 2)
    q["r_foot_orientation"] = (1 - np.inner(foot_orient_target, feet_pose[1, 3:]) ** 2)
    base_orient_target = np.array([1, 0, 0, 0])
    q["base_orientation"] = 1 - np.inner(base_pose[3:], base_orient_target) ** 2

    ### Static rewards. Want feet and motor velocities to be zero ###
    motor_vel = self.sim.get_motor_velocity()
    q['motor_vel_penalty'] = np.linalg.norm(motor_vel) / len(motor_vel)
    q['foot_vel_penalty'] = np.sum(feet_vel)

    ### Control rewards ###
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

# Termination condition: If height is too low (cassie fell down) terminate
def compute_done(self):
    base_height = self.sim.get_body_pose(self.sim.base_body_name)[2]
    if base_height < 0.4:
        return True
    else:
        return False
