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
    assert self.clock_type == "von_mises", \
        f"{FAIL}locomotion_vonmises_clock_reward should be used with von mises clock type, but " \
        f"clock type is {self.clock_type}.{ENDC}"

    q = {}

    ### Cyclic foot force/velocity reward ###
    l_force, r_force = self.clock.von_mises()
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

    q["left_force"] = l_force * np.abs(feet_force["left foot"])
    q["right_force"] = r_force * np.abs(feet_force["right foot"])
    q["left_speed"] = l_stance * feet_vel["left foot"]
    q["right_speed"] = r_stance * feet_vel["right foot"]

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
    if self.orient_add != 0:
        command_quat = euler2quat(z = self.orient_add, y = 0, x = 0)
        target_quat = quaternion_product(target_quat, command_quat)
    orientation_error = quaternion_distance(base_pose[3:], target_quat)
    # Deadzone around quaternion as well
    if orientation_error < 5e-3:
        orientation_error = 0

    # Foor orientation target in global frame. Want to be flat and face same direction as base all
    # the time. So compare to the same orientation target as the base.
    foot_orientation_error = quaternion_distance(target_quat, feet_pose["left foot"][3:]) + \
                             quaternion_distance(target_quat, feet_pose["right foot"][3:])
    q["orientation"] = orientation_error + foot_orientation_error

    ### Hop symmetry reward (keep feet equidistant) ###
    period_shifts = self.clock.get_period_shifts()
    rel_foot_pos = {}
    for foot in feet_pose:
        rel_foot_pos[foot] = (feet_pose[foot][0:3] - base_pose[0:3])[[0, 2]]
    xdif = np.sqrt(np.power(rel_foot_pos["left foot"] - rel_foot_pos["right foot"], 2).sum())
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
    orientation_error = 3 * quaternion_distance(base_quat, target_quat)
    if np.exp(-orientation_error) < 0.8:
        return True
    else:
        return False
