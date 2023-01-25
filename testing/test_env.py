import numpy as np

from env import (
    CassieEnv,
    CassieEnvClock,
    DigitEnv,
    DigitEnvClock
)

def test():
	test_cassieenv()
	test_cassieenvclock()
	test_digitenv()
	test_digitenvclock()

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
