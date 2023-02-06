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
    reward_list = [["linear", "locomotion_linear_clock_reward"],
                   ["von_mises", "locomotion_vonmises_clock_reward"],
                   ["linear", "stand_reward"]]

    for pair in base_env_sim_pair:
        test_base_env_step(test_env=pair[0], test_sim=pair[1])
        print(f"Pass test with {pair[0].__name__} and {pair[1]}.")

    for pair in child_env_list:
        test_child_env_step(test_env=pair[0], test_sim=pair[1])
        print(f"Pass test with {pair[0].__name__} and {pair[1]}.")

    for pair in child_env_list:
        for rew_pair in reward_list:
            test_child_env_reward(pair[0], pair[1], rew_pair[0], rew_pair[1])
            print(f"Pass test with {pair[0].__name__} and {pair[1]}, clock {rew_pair[0]}, and " \
                  f"reward {rew_pair[1]}.")

def test_base_env_step(test_env, test_sim):
    """Test if base env is step simulation in correct rate based on policy rate
    """
    env = test_env(simulator_type=test_sim,
                   policy_rate=50,
                   dynamics_randomization=False,
                   terrain=False)
    env.reset_simulation()
    sim_duration = []
    for i in range(100):
        start = env.sim.get_simulation_time()
        env.step_simulation(action=np.zeros(env.sim.num_actuators),
                                simulator_repeat_steps=int(env.sim.simulator_rate/env.default_policy_rate))
        sim_duration.append(env.sim.get_simulation_time() - start)
    assert np.abs(1 / env.default_policy_rate - np.mean(sim_duration)) < 1e-5,\
           f"Simulator steps by {np.mean(sim_duration)},"\
           f"but defined to step as {1 / env.default_policy_rate}"

def test_child_env_step(test_env, test_sim):
    """Test if child env is stepping based on specified policy rate.
    """
    env = test_env(cycle_time = 1.0,
                   simulator_type=test_sim,
                   policy_rate=50,
                   dynamics_randomization=False,
                   terrain=False,
                   clock_type="linear",
                   reward_name="locomotion_linear_clock_reward")
    env.reset()
    sim_duration = []
    for i in range(100):
        start = env.sim.get_simulation_time()
        s, r, _, _ = env.step(action=np.zeros(env.sim.num_actuators))
        assert None not in s, "Child env.step() returns state has None."
        assert r is not None, "Child env.step() returns reward as None."
        sim_duration.append(env.sim.get_simulation_time() - start)
    assert np.abs(1 / env.default_policy_rate - np.mean(sim_duration)) < 1e-5,\
           f"Simulator steps by {np.mean(sim_duration)},"\
           f"but defined to step as {1 / env.default_policy_rate}"

def test_child_env_reward(test_env, test_sim, clock_type, reward):
    """Test if child env with reward to make sure doesn't crash.
    """
    env = test_env(cycle_time = 1.0,
                   simulator_type=test_sim,
                   policy_rate=50,
                   dynamics_randomization=False,
                   terrain=False,
                   clock_type=clock_type,
                   reward_name=reward)
    env.reset()
    for i in range(100):
        s, r, _, _ = env.step(action=np.zeros(env.sim.num_actuators))
        assert r >= 0, \
            f"Env {test_env} with {test_sim}, clock {clock_type}, and reward {reward} encountered " \
            f"a negative reward."
        assert r <= 1, \
            f"Env {test_env} with {test_sim}, clock {clock_type}, and reward {reward} encountered " \
            f"a reward greater than 1."



