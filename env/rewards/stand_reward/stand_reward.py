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
    feet_vel = {}
    feet_pose = {}
    for foot_name in self.sim.feet_body_name:
        vel = np.linalg.norm(self.sim.get_body_velocity(foot_name)[0:3])
        side = "left" if "left" in foot_name else "right"
        feet_vel[f"{side} foot"] = vel
    for foot_name in self.sim.feet_site_name:
        pose = self.sim.get_site_pose(foot_name)
        side = "left" if "left" in foot_name else "right"
        feet_pose[f"{side} foot"] = pose

    orient_target = np.array([1, 0, 0, 0])
    q["base_orientation"] = quaternion_distance(base_pose[3:], orient_target)
    q["l_foot_orientation"] = quaternion_distance(orient_target, feet_pose["left foot"][3:])
    q["r_foot_orientation"] = quaternion_distance(orient_target, feet_pose["right foot"][3:])

    ### Static rewards. Want feet and motor velocities to be zero ###
    motor_vel = self.sim.get_motor_velocity()
    q['motor_vel_penalty'] = np.linalg.norm(motor_vel) / len(motor_vel)
    q['foot_vel_penalty'] = feet_vel["left foot"] + feet_vel["right foot"]

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
