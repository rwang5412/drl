import numpy as np

from env.genericenv import GenericEnv
from util.quaternion import quaternion_distance


def compute_rewards(self: GenericEnv, action):
    q = {}

    ### Height penalty, match the desired standing height ###
    base_pose = self.sim.get_body_pose(self.sim.base_body_name)
    q['height_penalty'] = np.abs(base_pose[2] - 0.9)

    ### Orientation rewards, base and feet ###
    # Retrieve states
    l_foot_vel = np.linalg.norm(self.feet_velocity_tracker_avg[self.sim.feet_body_name[0]])
    r_foot_vel = np.linalg.norm(self.feet_velocity_tracker_avg[self.sim.feet_body_name[1]])
    l_foot_pose = self.sim.get_site_pose(self.sim.feet_site_name[0])
    r_foot_pose = self.sim.get_site_pose(self.sim.feet_site_name[1])

    orient_target = np.array([1, 0, 0, 0])
    q["base_orientation"] = quaternion_distance(base_pose[3:], orient_target)
    q["l_foot_orientation"] = quaternion_distance(orient_target, l_foot_pose[3:])
    q["r_foot_orientation"] = quaternion_distance(orient_target, r_foot_pose[3:])

    ### Static rewards. Want feet and motor velocities to be zero ###
    motor_vel = self.sim.get_motor_velocity()
    q['motor_vel_penalty'] = np.linalg.norm(motor_vel) / len(motor_vel)
    q['foot_vel_penalty'] = l_foot_vel + r_foot_vel

    ### Control rewards ###
    if self.last_action is not None:
        q["ctrl_penalty"] = sum(np.abs(self.last_action - action)) / len(action)
    else:
        q["ctrl_penalty"] = 0
    torque = self.sim.get_torque()
    q["trq_penalty"] = sum(np.abs(torque)) / len(torque)

    return q

# Termination condition: If height is too low (cassie fell down) terminate
def compute_done(self: GenericEnv):
    base_height = self.sim.get_body_pose(self.sim.base_body_name)[2]
    if base_height < 0.4:
        return True
    else:
        return False
