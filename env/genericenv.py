
class GenericEnv(object):

    """
    Define generic environment functions that are needed for RL. Should define (not implement) all 
    of the functions that sampling uses.
    """

    def __init__(self):
        self.observation_size = None
        self.action_size = None

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
    
    def get_action_mirror_indices(self):
        raise NotImplementedError

    def get_observation_mirror_indices(self):
        raise NotImplementedError
