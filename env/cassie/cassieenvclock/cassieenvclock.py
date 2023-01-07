from rewards.blah import foo_reward

# Handles all 50Hz (policy rate) stuff
class CassieEnvClock(CassieEnv):

    def __init__(self, policy_rate):
        self.policy_rate = policy_rate
        if clock:
            self.clock = PeriodicTracker()
        
    def get_state():
        self.sim.get_joint_pos
        self.clock.input_clock

    def step(self, cmd):
        for i in range(self.policy_rate):
            self._sim_forward(cmd)

    def compute_reward(self):
        return foo_reward(self)