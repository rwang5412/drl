import numpy as np

from env import (
    CassieEnv,
    CassieEnvClock,
    DigitEnv,
    DigitEnvClock
)

def test():
	# test_cassieenv()
	# test_cassieenvclock()
	# test_digitenv()
	# test_digitenvclock()
	test_base_env_step()
	test_child_env_step()

def test_base_env_step():
	"""Test if base env is stepping based on policy rate
	"""
	env = CassieEnv(simulator_type="mujoco",
                	policy_rate=50,
                	dynamics_randomization=False,
                	terrain=False)
	env.reset_simulation()
	env.sim.viewer_init()
	sim_duration = []
	while env.sim.viewer.is_alive:
		if not env.sim.viewer.paused:
			start = env.sim.data.time
			env.step_simulation(action=np.zeros(env.sim.num_actuators), 
                       			simulator_repeat_steps=int(env.sim.simulator_rate/env.default_policy_rate))
			sim_duration.append(env.sim.data.time - start)
		env.sim.viewer_render()
	assert np.abs(1 / env.default_policy_rate - np.mean(sim_duration)) < 1e-5,\
		   f"Simulator steps by {np.mean(sim_duration)},"\
		   f"but defined to step as {1 / env.default_policy_rate}"

def test_child_env_step():
	"""Test if child env is stepping based on specified policy rate.
	"""
	env = CassieEnvClock(simulator_type="mujoco",
                		 policy_rate=50,
                		 dynamics_randomization=False,
                		 terrain=False,
                   		 clock=None,
                      	 reward_name=None)
	env.reset()
	env.sim.viewer_init()
	sim_duration = []
	while env.sim.viewer.is_alive:
		if not env.sim.viewer.paused:
			start = env.sim.data.time
			s, r, _, _ = env.step(action=np.zeros(env.sim.num_actuators))
			assert None not in s, "Child env.step() returns state has None."
			assert r is not None, "Child env.step() returns reward as None."
			sim_duration.append(env.sim.data.time - start)
		env.sim.viewer_render()
	assert np.abs(1 / env.default_policy_rate - np.mean(sim_duration)) < 1e-5,\
		   f"Simulator steps by {np.mean(sim_duration)},"\
		   f"but defined to step as {1 / env.default_policy_rate}"

def test_cassieenv():
	env = CassieEnv(simulator_type="mujoco",
                	policy_rate=50,
                	dynamics_randomization=False,
                	terrain=False)
	env.reset_simulation()
	env.sim.viewer_init()
	while env.sim.viewer.is_alive:
		if not env.sim.viewer.paused:
			env.step_simulation(action=np.zeros(10), simulator_repeat_steps=50)
		env.sim.viewer_render()

def test_digitenv():
	env = DigitEnv(simulator_type="mujoco",
                	policy_rate=50,
                	dynamics_randomization=False,
                	terrain=False)
	env.reset_simulation()
	env.sim.viewer_init()
	while env.sim.viewer.is_alive:
		if not env.sim.viewer.paused:
			env.step_simulation(action=np.zeros(20), simulator_repeat_steps=50)
		env.sim.viewer_render()

def test_cassieenvclock():
	env = CassieEnvClock(clock=None,
                      reward_name=None,
     				simulator_type="mujoco",
                	policy_rate=50,
                	dynamics_randomization=False,
                	terrain=False)
	s = env.reset()
	env.sim.viewer_init()
	while env.sim.viewer.is_alive:
		if not env.sim.viewer.paused:
			s, r, _, _ = env.step(action=np.zeros(10))
		env.sim.viewer_render()

def test_digitenvclock():
	env = DigitEnvClock(clock=None,
                      reward_name=None,
     				simulator_type="mujoco",
                	policy_rate=50,
                	dynamics_randomization=False,
                	terrain=False)
	s = env.reset()
	env.sim.viewer_init()
	while env.sim.viewer.is_alive:
		if not env.sim.viewer.paused:
			s, r, _, _ = env.step(action=np.zeros(20))
		env.sim.viewer_render()
