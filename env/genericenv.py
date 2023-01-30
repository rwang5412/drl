
class GenericEnv(object):

    """
    Define generic environment functions that are needed for RL. Should define (not implement) all of the functions that ppo.py sampling uses.
    """

    def __init__(self,
                 policy_rate: int,
                 dynamics_randomization: bool):

        self.dynamics_randomization = dynamics_randomization
        self.default_policy_rate = policy_rate

    def reset_simulation(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def step_simulation(self, action):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError

    def get_state(self):
        raise NotImplementedError

    def compute_reward(self):
        raise NotImplementedError
