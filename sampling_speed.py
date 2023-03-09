import mujoco as mj
import numpy as np
import time
from types import SimpleNamespace

from sim import (
    MjCassieSim,
    LibCassieSim,
)
from util.env_factory import env_factory

num_steps = 30000
# Test raw sampling speeds
sim_types = [MjCassieSim, LibCassieSim]
sim_speeds = {}
for sim_type in sim_types:
    sim = sim_type()
    total_time = 0
    for i in range(num_steps):
        sim.set_torque(np.random.uniform(size=(10,)))
        start_t = time.time()
        sim.sim_forward()
        total_time += time.time() - start_t
    sim_speeds[sim.__class__.__name__] = total_time / num_steps

for key, val in sim_speeds.items():
    print(f"Sim {key} averaged {val:.4e} seconds per step, running at a rate of {1 / val:.2f} Hz")

args = SimpleNamespace(simulator_type = "mujoco",
                       clock_type = "linear",
                       reward_name = "locomotion_linear_clock_reward",
                       dynamics_randomization = False)
mj_env = env_factory("CassieEnvClock", args)()
args = SimpleNamespace(simulator_type = "libcassie",
                       clock_type = "linear",
                       reward_name = "locomotion_linear_clock_reward",
                       dynamics_randomization = False)
lib_env = env_factory("CassieEnvClock", args)()

env_steps = 1000#num_steps // int(mj_env.sim.simulator_rate / mj_env.policy_rate)
# if env_steps != num_steps / int(mj_env.sim.simulator_rate / mj_env.policy_rate):
    # print("Policy rate does not fit evenly into the selected number of sim test steps")
    # exit(1)
total_time = 0
mj_sim_per_step = int(mj_env.sim.simulator_rate / mj_env.default_policy_rate)
# mj_env.trackers = []
for i in range(env_steps):
    act = np.random.uniform(size=(10,))
    start_t = time.time()
    mj_env.step(act)
    total_time += time.time() - start_t
mj_avg_time = total_time / (env_steps * mj_sim_per_step)
mj_overhead = (mj_avg_time - sim_speeds["MjCassieSim"]) * mj_sim_per_step

total_time = 0
lib_sim_per_step = int(lib_env.sim.simulator_rate / lib_env.default_policy_rate)
for i in range(env_steps):
    act = np.random.uniform(size=(10,))
    start_t = time.time()
    lib_env.step(act)
    total_time += time.time() - start_t
lib_avg_time = total_time / (env_steps * lib_sim_per_step)
lib_overhead = (lib_avg_time - sim_speeds["LibCassieSim"]) * lib_sim_per_step

print(f"Env with mujoco averaged {mj_avg_time:.4e}, running at rate of {1 / mj_avg_time:.2f} Hz.")
print(f"Lose {mj_overhead:.4e} seconds per env step due to env overhead. Over 500 million steps " \
      f"lose {mj_overhead * 500000000/60/60/24:.2f} days")

print(f"Env with libcassie averaged {lib_avg_time:.4e}, running at rate of {1 / lib_avg_time:.2f} Hz.")
print(f"Lose {lib_overhead:.4e} seconds per env step due to env overhead. Over 500 million steps " \
      f"lose {lib_overhead * 500000000/60/60/24:.2f} days")


