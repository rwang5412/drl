import numpy as np

from env import (
	CassieEnv,
	DigitEnv,
	CassieEnvClock,
	DigitEnvClock
)

def test_all_env():
	base_env_sim_pair = [[CassieEnv, "mujoco"], [DigitEnv, "mujoco"],
                      	 [CassieEnv, "libcassie"]]
	child_env_list = [[CassieEnvClock, "mujoco"], [DigitEnvClock, "mujoco"],
                	  [CassieEnvClock, "libcassie"]]

	for pair in base_env_sim_pair:
		test_base_env_step(test_env=pair[0], test_sim=pair[1])
		print(f"Pass test with {pair[0].__name__} and {pair[1]}.")

	for pair in child_env_list:
		test_child_env_step(test_env=pair[0], test_sim=pair[1])
		print(f"Pass test with {pair[0].__name__} and {pair[1]}.")

def test_base_env_step(test_env, test_sim):
	"""Test if base env is step simulation in correct rate based on policy rate
	"""
	env = test_env(simulator_type=test_sim,
				   policy_rate=50,
				   dynamics_randomization=False,
				   terrain=False)
	env.reset_simulation()
	env.sim.viewer_init()
	sim_duration = []
	render_state = env.sim.viewer_render()
	while render_state:
		if not env.sim.viewer_paused():
			start = env.sim.get_simulation_time()
			env.step_simulation(action=np.zeros(env.sim.num_actuators),
					   			simulator_repeat_steps=int(env.sim.simulator_rate/env.default_policy_rate))
			sim_duration.append(env.sim.get_simulation_time() - start)
		render_state = env.sim.viewer_render()
	assert np.abs(1 / env.default_policy_rate - np.mean(sim_duration)) < 1e-5,\
		   f"Simulator steps by {np.mean(sim_duration)},"\
		   f"but defined to step as {1 / env.default_policy_rate}"

def test_child_env_step(test_env, test_sim):
	"""Test if child env is stepping based on specified policy rate.
	"""
	env = test_env(simulator_type=test_sim,
				   policy_rate=50,
				   dynamics_randomization=False,
				   terrain=False,
				   clock=None,
				   reward_name=None)
	env.reset()
	env.sim.viewer_init()
	sim_duration = []
	render_state = env.sim.viewer_render()
	while render_state:
		if not env.sim.viewer_paused():
			start = env.sim.get_simulation_time()
			s, r, _, _ = env.step(action=np.zeros(env.sim.num_actuators))
			assert None not in s, "Child env.step() returns state has None."
			assert r is not None, "Child env.step() returns reward as None."
			sim_duration.append(env.sim.get_simulation_time() - start)
		render_state = env.sim.viewer_render()
	assert np.abs(1 / env.default_policy_rate - np.mean(sim_duration)) < 1e-5,\
		   f"Simulator steps by {np.mean(sim_duration)},"\
		   f"but defined to step as {1 / env.default_policy_rate}"
