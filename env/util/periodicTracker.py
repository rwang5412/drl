
class PeriodicTracker:
    """
    A class defines clock that increments internaly, along with getter/setter to change clocks.
    """

    def __init__(clock_type):
        pass
    
    def input_clock(self, phase, phaselen):
        pass

    def reward_clock(self, phase, phaselen, swing_ratio):
        pass

    def phase_increment(self):
        pass

class linearClock(PeriodicTracker):
    pass

class vonMisesclock(PeriodicTracker):
    pass