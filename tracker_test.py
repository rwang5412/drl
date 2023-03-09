import mujoco as mj
import numpy as np
import time
from types import SimpleNamespace
import torch
import pickle
import os
import matplotlib.pyplot as plt

from util.env_factory import env_factory
from util.nn_factory import nn_factory, load_checkpoint
from env.util.periodicclock import PeriodicClock

args = SimpleNamespace(simulator_type = "mujoco",
                       clock_type = "linear",
                       reward_name = "locomotion_linear_clock_reward",
                       dynamics_randomization = False)
env = env_factory("CassieEnvClock", args)()

model_path = "./pretrained_models/CassieEnvClockOld/"
previous_args_dict = pickle.load(open(os.path.join(model_path, "experiment.pkl"), "rb"))
actor_checkpoint = torch.load(os.path.join(model_path, 'actor.pt'), map_location='cpu')
previous_args_dict['env_args'].dynamics_randomization = False

# Load environment
env = env_factory(previous_args_dict['all_args'].env_name, previous_args_dict['env_args'])()

# Load model class and checkpoint
actor, critic = nn_factory(args=previous_args_dict['nn_args'], env=env)
load_checkpoint(model=actor, model_dict=actor_checkpoint)
actor.eval()
actor.training = False

env_steps = 200
grfs = []
foot_vel = []
state = env.reset()
env.x_velocity = 1
env.y_velocity = 0
env.turn_rate = 0
env.clock = PeriodicClock(0.8, 1 / env.default_policy_rate, [0.5, 0.5], [0, 0.5])
np.random.seed(8888)

with torch.no_grad():
    if hasattr(actor, 'init_hidden_state'):
        actor.init_hidden_state()
    for i in range(env_steps):
        # state = torch.Tensor(state).float()
        # action = actor(state).numpy()
        action = np.random.uniform(-0.5, 0.5, env.action_size)
        state, reward, done, _ = env.step(action)
        grfs.append([env.feet_grf_2khz_avg["left-foot"], env.feet_grf_2khz_avg["right-foot"]])
        foot_vel.append([env.feet_velocity_2khz_avg["left-foot"], env.feet_velocity_2khz_avg["right-foot"]])

grfs = np.array(grfs)
foot_vel = np.array(foot_vel)
norm_foot_vel = np.linalg.norm(foot_vel[:, :, 0:3], axis=2)
print(norm_foot_vel.shape)
print(grfs.shape, foot_vel.shape)
fig, ax = plt.subplots(2, 1, figsize=(15, 6))
t = np.linspace(0, env_steps * 1 / env.default_policy_rate, env_steps)
ax[0].plot(t, grfs[:, 0, 2])
ax[0].plot(t, grfs[:, 1, 2])
ax[1].plot(t, norm_foot_vel[:, 0])
ax[1].plot(t, norm_foot_vel[:, 1])

# plt.show()
plt.savefig("./50Hz_tracker_random.png")


