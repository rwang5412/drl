import matplotlib.pyplot as plt
import numpy as np
import time
from env import (
    CassieEnvClock,
    DigitEnvClock
)
from env.util.periodicclock import PeriodicClock


def plot_clock(t, lr_swing_vals, title: str):
    # Input should be 2d array of shape [swing values, 2], left will come first ([:, 0]) and right
    # is second ([:, 1])
    fig, ax = plt.subplots(2, 1, figsize=(8,5))
    ax[0].plot(t, lr_swing_vals[:, 0], label="swing")
    ax[0].plot(t, 1 - lr_swing_vals[:, 0], label="stance")
    ax[1].plot(t, lr_swing_vals[:, 1], label="swing")
    ax[1].plot(t, 1 - lr_swing_vals[:, 1], label="stance")

    ax[0].set_title("Left Foot", fontsize=18)
    ax[1].set_title("Right Foot", fontsize=18)
    ax[1].set_xlabel("Time (sec)", fontsize=16)
    ax[0].set_ylabel("Cost Weighting", fontsize=16)
    ax[1].set_ylabel("Cost Weighting", fontsize=16)
    fig.suptitle(title, fontsize=20)

    ax[0].legend(loc=1, prop={"size":16})
    ax[1].legend(loc=1, prop={"size":16})
    ax[0].tick_params(axis='both', which='major', labelsize=16)
    ax[1].tick_params(axis='both', which='major', labelsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def test_all_clocks():
    test_linear_walk_clock()
    test_linear_run_clock()
    test_linear_hop_clock()
    test_linear_gallop_clock()

    test_vonmises_walk_clock()
    test_vonmises_run_clock()
    test_vonmises_hop_clock()
    test_vonmises_gallop_clock()
    test_vonmises_precompute()
    test_cassieclockenv()

def test_linear_walk_clock():
    phaselen = 1
    swing_ratios = np.array([0.4, 0.4])
    period_shifts = np.array([0, 0.5])
    tracker = PeriodicClock(phaselen, 1 / 50, swing_ratios, period_shifts)
    xs = np.linspace(0, phaselen, 1000)
    ys = np.zeros((1000, 2))
    for i in range(len(xs)):
        tracker.set_phase(xs[i])
        ys[i, :] = tracker.linear_clock()
    plot_clock(xs, ys, "Walking Clock")

def test_linear_run_clock():
    phaselen = 1
    swing_ratios = np.array([0.8, 0.8])
    period_shifts = np.array([0, 0.5])
    tracker = PeriodicClock(phaselen, 1 / 50, swing_ratios, period_shifts)
    xs = np.linspace(0, phaselen, 1000)
    ys = np.zeros((1000, 2))
    for i in range(len(xs)):
        tracker.set_phase(xs[i])
        ys[i, :] = tracker.linear_clock()
    plot_clock(xs, ys, "Running Clock")

def test_linear_hop_clock():
    phaselen = 1
    swing_ratios = np.array([0.5, 0.5])
    period_shifts = np.array([0, 0])
    tracker = PeriodicClock(phaselen, 1 / 50, swing_ratios, period_shifts)
    xs = np.linspace(0, phaselen, 1000)
    ys = np.zeros((1000, 2))
    for i in range(len(xs)):
        tracker.set_phase(xs[i])
        ys[i, :] = tracker.linear_clock()
    plot_clock(xs, ys, "Hopping Clock")

def test_linear_gallop_clock():
    phaselen = 1
    swing_ratios = np.array([0.6, 0.6])
    period_shifts = np.array([0, 0.25])
    tracker = PeriodicClock(phaselen, 1 / 50, swing_ratios, period_shifts)
    xs = np.linspace(0, phaselen, 1000)
    ys = np.zeros((1000, 2))
    for i in range(len(xs)):
        tracker.set_phase(xs[i])
        ys[i, :] = tracker.linear_clock()
    plot_clock(xs, ys, "Uneven Gallop Clock")

def test_vonmises_walk_clock():
    phaselen = 1
    swing_ratios = np.array([0.4, 0.4])
    period_shifts = np.array([0.0, 0.5])
    tracker = PeriodicClock(phaselen, 1 / 50, swing_ratios, period_shifts)
    xs = np.linspace(0, phaselen, 1000)
    ys = np.zeros((1000, 2))
    for i in range(len(xs)):
        tracker.set_phase(xs[i])
        ys[i, :] = tracker.von_mises()
    plot_clock(xs, ys, "Von Mises Walk Clock")

def test_vonmises_run_clock():
    phaselen = 1
    swing_ratios = np.array([0.8, 0.8])
    period_shifts = np.array([0.0, 0.5])
    tracker = PeriodicClock(phaselen, 1 / 50, swing_ratios, period_shifts)
    xs = np.linspace(0, phaselen, 1000)
    ys = np.zeros((1000, 2))
    for i in range(len(xs)):
        tracker.set_phase(xs[i])
        ys[i, :] = tracker.von_mises()
    plot_clock(xs, ys, "Von Mises Run Clock")

def test_vonmises_hop_clock():
    phaselen = 1
    swing_ratios = np.array([0.5, 0.5])
    period_shifts = np.array([0.0, 0.0])
    tracker = PeriodicClock(phaselen, 1 / 50, swing_ratios, period_shifts)
    xs = np.linspace(0, phaselen, 1000)
    ys = np.zeros((1000, 2))
    for i in range(len(xs)):
        tracker.set_phase(xs[i])
        ys[i, :] = tracker.von_mises()
    plot_clock(xs, ys, "Von Mises Hop Clock")

def test_vonmises_gallop_clock():
    phaselen = 1
    swing_ratios = np.array([0.6, 0.6])
    period_shifts = np.array([0.0, 0.25])
    tracker = PeriodicClock(phaselen, 1 / 50, swing_ratios, period_shifts)
    xs = np.linspace(0, phaselen, 1000)
    ys = np.zeros((1000, 2))
    for i in range(len(xs)):
        tracker.set_phase(xs[i])
        ys[i, :] = tracker.von_mises()
    plot_clock(xs, ys, "Von Mises Gallop Clock")

def test_vonmises_precompute():
    phaselen = 1
    swing_ratios = np.array([0.4, 0.4])
    period_shifts = np.array([0.0, 0.5])
    tracker = PeriodicClock(phaselen, 1 / 50, swing_ratios, period_shifts)
    xs = np.linspace(0, phaselen, 1000)
    ys = np.zeros((1000, 2))
    start_t = time.time()
    for i in range(len(xs)):
        tracker.set_phase(xs[i])
        ys[i, :] = tracker.von_mises()
    print("Von mises time", time.time() - start_t)
    tracker.precompute_von_mises(num_points = 1000)
    start_t = time.time()
    ys = np.zeros((1000, 2))
    for i in range(len(xs)):
        tracker.set_phase(xs[i])
        ys[i, :] = tracker.get_von_mises_values()
    print("Precompute time", time.time() - start_t)

def test_cassieclockenv():
    cycle_time = 1.0
    clock_type = "linear"
    policy_rate = 50
    env = CassieEnvClock(cycle_time = cycle_time,
                         clock_type = clock_type,
                         reward_name = "foo",
                         simulator_type = "mujoco",
                         terrain = False,
                         policy_rate = policy_rate,
                         dynamics_randomization = False)
    for i in range(int(cycle_time * policy_rate)):
        env.step(np.zeros(10))
    assert env.clock.get_phase() < 0.001, \
        f"Failed CassieClockEnv linear test, after one cycle phase should be 0, but phase is " \
        f"{env.clock.get_phase()}"
    clock_type = "von_mises"
    env = CassieEnvClock(cycle_time = cycle_time,
                         clock_type = clock_type,
                         reward_name = "foo",
                         simulator_type = "mujoco",
                         terrain = False,
                         policy_rate = policy_rate,
                         dynamics_randomization = False)
    for i in range(int(cycle_time * policy_rate)):
        env.step(np.zeros(10))
    assert env.clock.get_phase() < 0.001, \
        f"Failed CassieClockEnv von mises test, after one cycle phase should be 0, but phase is " \
        f"{env.clock.get_phase()}"

def test_digitclockenv():
    cycle_time = 1.0
    clock_type = "linear"
    policy_rate = 50
    env = DigitEnvClock(cycle_time = cycle_time,
                         clock_type = clock_type,
                         reward_name = "foo",
                         simulator_type = "mujoco",
                         terrain = False,
                         policy_rate = policy_rate,
                         dynamics_randomization = False)
    for i in range(int(cycle_time * policy_rate)):
        env.step(np.zeros(10))
    assert env.clock.get_phase() < 0.001, \
        f"Failed CassieClockEnv linear test, after one cycle phase should be 0, but phase is " \
        f"{env.clock.get_phase()}"
    clock_type = "von_mises"
    env = DigitEnvClock(cycle_time = cycle_time,
                         clock_type = clock_type,
                         reward_name = "foo",
                         simulator_type = "mujoco",
                         terrain = False,
                         policy_rate = policy_rate,
                         dynamics_randomization = False)
    for i in range(int(cycle_time * policy_rate)):
        env.step(np.zeros(10))
    assert env.clock.get_phase() < 0.001, \
        f"Failed CassieClockEnv von mises test, after one cycle phase should be 0, but phase is " \
        f"{env.clock.get_phase()}"