import numpy as np
import matplotlib.pyplot as plt
from env.PeriodicTracker import PeriodicTracker

def test_linear_clock():
    phaselen = 1
    swing_ratios = np.array([0.4, 0.4])
    period_shifts = np.array([0, 0.5])
    tracker = PeriodicTracker(phaselen, swing_ratios, period_shifts)
    xs = np.linspace(0, phaselen, 1000)
    ys = np.zeros(1000)
    ys = np.array(list(map(tracker.linear_clock, xs)))
    print(ys.shape)

    fig, ax = plt.subplots(2, 1, figsize=(8,5))
    ax[0].plot(xs, ys[:, 0], label="swing")
    # ax[0].plot(t, l_stance, label="stance")
    ax[1].plot(xs, ys[:, 1], label="swing")
    # ax[1].plot(t, r_stance, label="stance")
    plt.show()