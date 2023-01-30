import numpy as np
from scipy.stats import vonmises
from typing import List

# I'm actually not sure if we want this to be a class or not. We could make class and then have it
# hold on to the variables, or just make them all functions have things like phase, phaselen, etc.
# be inputs to the function.
class PeriodicClock:

    """
    Class for keeping track of clock values.
    """

    def __init__(self, cycle_time: float, swing_ratios: np.ndarray, period_shifts: np.ndarray):
        # Should be class variables or be pushed into the env itself?
        assert swing_ratios.shape == (2,), \
               f"set_joint_position got array of shape {swing_ratios.shape} but " \
               f"should be shape (2,)."
        assert period_shifts.shape == (2,), \
               f"set_joint_position got array of shape {period_shifts.shape} but " \
               f"should be shape (2,)."
        self._cycle_time = cycle_time
        # Assume that swing ratios and period shifts are in order of [left, right]
        self._swing_ratios = swing_ratios
        self._period_shifts = period_shifts
        self._von_mises_buf = None

    def input_clock(self, phase):
        # NOTE: This is doing straight sin/cos clock. I don't know if we want to use the input
        # clocks where it changes with period shift too.
        clock = [np.sin(2 * np.pi * (phase / self._phase_len)),
                 np.cos(2 * np.pi * (phase / self._phase_len))]
        return clock

    def linear_clock(self, phase, percent_transition: float = 0.2):
        y_clock = []
        x_clock = []
        phases = []
        for i in range(2):
            swing_time = self._cycle_time * self._swing_ratios[i]
            stance_time = self._cycle_time * (1 - self._swing_ratios[i])
            trans_time = swing_time * percent_transition
            swing_time -= trans_time
            y_clock.append([0, 0, 1, 1, 0])
            x_clock.append([0, stance_time, stance_time + trans_time / 2,
                    stance_time + trans_time / 2 + swing_time, self._cycle_time])
            if phase + self._period_shifts[i] > self._cycle_time:
                phases.append(phase + self._period_shifts[i] - self._cycle_time)
            else:
                phases.append(phase + self._period_shifts[i])
        return np.interp(phases[0], x_clock[0], y_clock[0]), np.interp(phases[1], x_clock[1], y_clock[1])


    def von_mises(self, phase, std: float = 0.1):
        # Von Mises clock function, but with hard coded coeff values of [0, 1]. This is chosen to
        # line up with the linear clock convention of left stance starting at t = 0. Because of this
        # coefficient choice, we actually need to "flip" the swing/stance ratio. If coeff was [1, 0]
        # then swing ratio would be as defined, i.e. if self._swing_ratio[0] = 0.4 then left foot
        # would have 40% swing, but left SWING would start at t = 0. Since we use coeff of [0, 1]
        # instead, swing ratio because stance ratio and vice versa.
        kappa = 1 / (std ** 2)

        out = []
        for i in range(2):
            x = (phase / self._cycle_time + self._period_shifts[i]) * 2 * np.pi
            # Use `1 - self._swing_ratios[i]` here to flip swing and stance ratios.
            mid = (1 - self._swing_ratios[i]) * 2 * np.pi
            end = 2 * np.pi
            p1 = vonmises.cdf(x, kappa=kappa, loc=0, scale=self._cycle_time)
            p2 = vonmises.cdf(x, kappa=kappa, loc=mid, scale=1)
            p3 = vonmises.cdf(x, kappa=kappa, loc=end, scale=1)
            out.append(p2 - p3)
            # out.append(p1 - p2)

        return out[0], out[1]

    def von_mises_full(self, phase, coeff: List[float] = [0.0, 1.0], std: float  = 0.2):
        # This is an alternate version of the von mises clock function. This will do the full
        # computation and allow for different coefficient values which will change the shape of the
        # clock. I think for most cases the coeffs we woudl use are [0, 1], which simplifies the
        # computation. In this case, use the regular `von_mises` function which should be faster.
        kappa = 1 / (std ** 2)

        out = []
        for i in range(2):
            x = (phase / self._cycle_time + self._period_shifts[i]) * 2 * np.pi
            time = 0
            start = 0
            mid = self._swing_ratios[i] * 2 * np.pi
            end = 2 * np.pi
            p1 = vonmises.cdf(x, kappa=kappa, loc=start, scale=self._cycle_time)
            p2 = vonmises.cdf(x, kappa=kappa, loc=mid, scale=self._cycle_time)
            p3 = vonmises.cdf(x, kappa=kappa, loc=end, scale=self._cycle_time)
            out.append(coeff[0] * (p1 - p2) + coeff[1] * (p2 - p3))

        return out[0], out[1]

    def precompute_von_mises(self, num_points: int = 200, std = 0.1):
        # Von mises clock can take a while to compute since it has to compute the cdf multiple times
        # so during training for an env it can be useful to precompute the clock values just once
        # beforehand. Note that this function has to be called again if _swing_ratios or
        # _period_shifts change, the values have to be recomputed
        xs = np.linspace(0, self._cycle_time, num_points)
        self._von_mises_buf = np.array(list(map(self.von_mises, xs)))

    def get_von_mises_values(self, phase):
        assert self._von_mises_buf is not None, \
            f"Von Mises clock buffer is None, can not get value. Call `precompute_von_mises` first."
        xs = np.linspace(0, self._cycle_time, self._von_mises_buf.shape[0])
        return np.interp(phase, xs, self._von_mises_buf[:, 0]), np.interp(phase, xs, self._von_mises_buf[:, 1])

